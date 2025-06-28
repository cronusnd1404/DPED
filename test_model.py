# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

import imageio
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys

tf.compat.v1.disable_v2_behavior()

# process command arguments
phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# Configure GPU memory for inference
if use_gpu == "true":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # Reduce memory usage
    config.allow_soft_placement = True
else:
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

# Define patch size for processing large images
PATCH_SIZE_PROC = 512  # Process in 512x512 patches
OVERLAP = 64  # Overlap between patches for seamless blending

def process_image_in_patches(sess, enhanced, x_, image, patch_size=PATCH_SIZE_PROC, overlap=OVERLAP):
    """Process large image in overlapping patches to avoid memory issues"""
    h, w, c = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate stride and number of patches needed
    stride = patch_size - overlap
    
    # Calculate number of patches needed to cover the entire image
    h_patches = int(np.ceil((h - overlap) / stride))
    w_patches = int(np.ceil((w - overlap) / stride))
    
    print(f"Processing image {h}x{w} in {h_patches}x{w_patches} patches of size {patch_size}x{patch_size}")
    
    for i in range(h_patches):
        for j in range(w_patches):
            # Calculate patch boundaries
            start_h = i * stride
            start_w = j * stride
            
            # Ensure we don't exceed image boundaries
            end_h = min(start_h + patch_size, h)
            end_w = min(start_w + patch_size, w)
            
            # If patch would be too small, adjust start position
            if end_h - start_h < patch_size:
                start_h = max(0, h - patch_size)
                end_h = h
            if end_w - start_w < patch_size:
                start_w = max(0, w - patch_size)
                end_w = w
            
            actual_patch_h = end_h - start_h
            actual_patch_w = end_w - start_w
            
            # Extract patch
            patch = image[start_h:end_h, start_w:end_w, :]
            
            # Pad patch to required size if needed
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
            
            # Process patch
            patch_2d = np.reshape(patch, [1, patch_size * patch_size * 3])
            enhanced_patch_2d = sess.run(enhanced, feed_dict={x_: patch_2d})
            enhanced_patch = np.reshape(enhanced_patch_2d, [patch_size, patch_size, 3])
            
            # Extract only the actual size from enhanced result
            enhanced_patch = enhanced_patch[:actual_patch_h, :actual_patch_w, :]
            
            # Create weight map for blending
            patch_weight = np.ones((actual_patch_h, actual_patch_w), dtype=np.float32)
            
            # Apply feathering at patch edges for smooth blending
            if overlap > 0:
                feather_size = min(overlap // 2, 32)  # Limit feather size
                
                # Create feathering mask
                for y in range(actual_patch_h):
                    for x in range(actual_patch_w):
                        # Distance from edges
                        dist_left = x
                        dist_right = actual_patch_w - 1 - x
                        dist_top = y
                        dist_bottom = actual_patch_h - 1 - y
                        
                        # Apply feathering only if we're at image boundaries that have neighbors
                        weight_factor = 1.0
                        
                        # Left edge feathering
                        if start_w > 0 and dist_left < feather_size:
                            weight_factor = min(weight_factor, dist_left / feather_size)
                        
                        # Right edge feathering
                        if end_w < w and dist_right < feather_size:
                            weight_factor = min(weight_factor, dist_right / feather_size)
                        
                        # Top edge feathering
                        if start_h > 0 and dist_top < feather_size:
                            weight_factor = min(weight_factor, dist_top / feather_size)
                        
                        # Bottom edge feathering
                        if end_h < h and dist_bottom < feather_size:
                            weight_factor = min(weight_factor, dist_bottom / feather_size)
                        
                        patch_weight[y, x] = weight_factor
            
            # Accumulate weighted results
            for ch in range(3):
                result[start_h:end_h, start_w:end_w, ch] += enhanced_patch[:, :, ch] * patch_weight
            weight_map[start_h:end_h, start_w:end_w] += patch_weight
            
            print(f"Processed patch {i+1}/{h_patches}, {j+1}/{w_patches}")
    
    # Normalize by weight map to get final result
    for ch in range(3):
        # Avoid division by zero
        mask = weight_map > 0
        result[:, :, ch][mask] = result[:, :, ch][mask] / weight_map[mask]
        # For areas with no weight, use original image
        result[:, :, ch][~mask] = image[:, :, ch][~mask]
    
    return result

# create placeholders for input images (use patch size)
x_ = tf.compat.v1.placeholder(tf.float32, [None, PATCH_SIZE_PROC * PATCH_SIZE_PROC * 3])
x_image = tf.reshape(x_, [-1, PATCH_SIZE_PROC, PATCH_SIZE_PROC, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:

    test_dir = dped_dir + phone.replace("_orig", "") + "/test_data/full_size_test_images/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models_orig/" + phone)

        for photo in test_photos:

            # load training image and crop it if necessary

            print("Testing original " + phone.replace("_orig", "") + " model, processing image " + photo)
            image = np.float32(np.array(Image.fromarray(imageio.imread(test_dir + photo))
                                        .resize([res_sizes[phone][1], res_sizes[phone][0]]))) / 255

            image_crop = utils.extract_crop(image, resolution, phone, res_sizes)

            # Process image in patches to avoid memory issues
            enhanced_image = process_image_in_patches(sess, enhanced, x_, image_crop)

            before_after = np.hstack((image_crop, enhanced_image))
            photo_name = photo.rsplit(".", 1)[0]

            # save the results as .png images
            # Convert to uint8 and clip values to valid range
            enhanced_image_uint8 = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
            before_after_uint8 = np.clip(before_after * 255, 0, 255).astype(np.uint8)

            imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_enhanced.png", enhanced_image_uint8)
            imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_before_after.png", before_after_uint8)

    else:

        num_saved_models = int(len([f for f in os.listdir("models/") if f.startswith(phone + "_iteration")]) / 2)

        if iteration == "all":
            iteration = np.arange(1, num_saved_models) * 1000
        else:
            iteration = [int(iteration)]

        for i in iteration:

            # load pre-trained model
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, "models/" + phone + "_iteration_" + str(i) + ".ckpt")

            for photo in test_photos:

                # load training image and crop it if necessary

                print("iteration " + str(i) + ", processing image " + photo)
                image = np.float32(np.array(Image.fromarray(imageio.imread(test_dir + photo))
                                            .resize([res_sizes[phone][1], res_sizes[phone][0]]))) / 255

                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)

                # Process image in patches to avoid memory issues
                enhanced_image = process_image_in_patches(sess, enhanced, x_, image_crop)

                before_after = np.hstack((image_crop, enhanced_image))
                photo_name = photo.rsplit(".", 1)[0]

                # save the results as .png images
                # Convert to uint8 and clip values to valid range
                enhanced_image_uint8 = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
                before_after_uint8 = np.clip(before_after * 255, 0, 255).astype(np.uint8)

                imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png", enhanced_image_uint8)
                imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png", before_after_uint8)
