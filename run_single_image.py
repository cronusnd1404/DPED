import imageio
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import argparse
import re

tf.compat.v1.disable_v2_behavior()

def process_image_in_patches(sess, enhanced, x_, image, patch_size=512, overlap=64):
    h, w, c = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    stride = patch_size - overlap
    h_patches = int(np.ceil((h - overlap) / stride))
    w_patches = int(np.ceil((w - overlap) / stride))
    for i in range(h_patches):
        for j in range(w_patches):
            start_h = i * stride
            start_w = j * stride
            end_h = min(start_h + patch_size, h)
            end_w = min(start_w + patch_size, w)
            if end_h - start_h < patch_size:
                start_h = max(0, h - patch_size)
                end_h = h
            if end_w - start_w < patch_size:
                start_w = max(0, w - patch_size)
                end_w = w
            actual_patch_h = end_h - start_h
            actual_patch_w = end_w - start_w
            patch = image[start_h:end_h, start_w:end_w, :]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
            patch_2d = np.reshape(patch, [1, patch_size * patch_size * 3])
            enhanced_patch_2d = sess.run(enhanced, feed_dict={x_: patch_2d})
            enhanced_patch = np.reshape(enhanced_patch_2d, [patch_size, patch_size, 3])
            enhanced_patch = enhanced_patch[:actual_patch_h, :actual_patch_w, :]
            patch_weight = np.ones((actual_patch_h, actual_patch_w), dtype=np.float32)
            if overlap > 0:
                feather_size = min(overlap // 2, 32)
                for y in range(actual_patch_h):
                    for x in range(actual_patch_w):
                        dist_left = x
                        dist_right = actual_patch_w - 1 - x
                        dist_top = y
                        dist_bottom = actual_patch_h - 1 - y
                        weight_factor = 1.0
                        if start_w > 0 and dist_left < feather_size:
                            weight_factor = min(weight_factor, dist_left / feather_size)
                        if end_w < w and dist_right < feather_size:
                            weight_factor = min(weight_factor, dist_right / feather_size)
                        if start_h > 0 and dist_top < feather_size:
                            weight_factor = min(weight_factor, dist_top / feather_size)
                        if end_h < h and dist_bottom < feather_size:
                            weight_factor = min(weight_factor, dist_bottom / feather_size)
                        patch_weight[y, x] = weight_factor
            for ch in range(3):
                result[start_h:end_h, start_w:end_w, ch] += enhanced_patch[:, :, ch] * patch_weight
            weight_map[start_h:end_h, start_w:end_w] += patch_weight
    for ch in range(3):
        mask = weight_map > 0
        result[:, :, ch][mask] = result[:, :, ch][mask] / weight_map[mask]
        result[:, :, ch][~mask] = image[:, :, ch][~mask]
    return result

def get_next_incremental_filename(folder, ext=".png"):
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(folder)
    numbers = []
    for f in files:
        m = re.match(r"(\d+)" + re.escape(ext) + r"$", f)
        if m:
            numbers.append(int(m.group(1)))
    next_num = 1 if not numbers else max(numbers) + 1
    return os.path.join(folder, f"{next_num}{ext}")

def main():
    parser = argparse.ArgumentParser(description='Enhance a single image using a DPED model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, choices=['iphone', 'sony', 'blackberry', 'iphone_orig', 'sony_orig', 'blackberry_orig'], help='Model type to use')
    parser.add_argument('--iteration', default='18000', help='Model iteration to use (default: 18000)')
    parser.add_argument('--output', help='Output path (default: auto-generated)')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for processing (default: 512)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--resize', help='Resize image to model resolution (e.g., "1024,768")')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found")
        return

    if args.use_gpu:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.allow_soft_placement = True
    else:
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    res_sizes = utils.get_resolutions()
    phone = args.model

    print(f"Loading image: {args.image}")
    try:
        image_raw = imageio.imread(args.image)
        if len(image_raw.shape) == 2:
            image_raw = np.stack([image_raw] * 3, axis=-1)
        elif image_raw.shape[2] == 4:
            image_raw = image_raw[:, :, :3]
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if args.resize:
        try:
            w, h = map(int, args.resize.split(','))
            image_raw = np.array(Image.fromarray(image_raw).resize([w, h]))
            print(f"Resized image to {w}x{h}")
        except Exception as e:
            print(f"Error resizing image: {e}")
            return
    elif phone in res_sizes:
        target_w, target_h = res_sizes[phone][1], res_sizes[phone][0]
        image_raw = np.array(Image.fromarray(image_raw).resize([target_w, target_h]))
        print(f"Resized image to model resolution: {target_w}x{target_h}")

    image = np.float32(image_raw) / 255.0
    patch_size = args.patch_size
    x_ = tf.compat.v1.placeholder(tf.float32, [None, patch_size * patch_size * 3])
    x_image = tf.reshape(x_, [-1, patch_size, patch_size, 3])
    enhanced = resnet(x_image)

    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.Saver()
        if phone.endswith("_orig"):
            model_path = f"models_orig/{phone}"
            if not os.path.exists(model_path + ".meta"):
                print(f"Error: Original model not found at {model_path}")
                return
            print(f"Loading original model: {model_path}")
            saver.restore(sess, model_path)
        else:
            model_dir = "models/"
            if not os.path.exists(model_dir):
                print(f"Error: Models directory not found: {model_dir}")
                return
            all_files = os.listdir(model_dir)
            # Look for .index files to find available checkpoints
            model_files = [f for f in all_files if f.startswith(f"{phone}_iteration") and f.endswith(".ckpt.index")]
            if not model_files:
                print(f"Error: No models found for {phone}")
                return
            if args.iteration == "latest":
                iterations = []
                for f in model_files:
                    try:
                        iter_num = int(f.split("_iteration_")[1].split(".ckpt.index")[0])
                        iterations.append(iter_num)
                    except:
                        continue
                if not iterations:
                    print("Error: Could not find any valid model iterations")
                    return
                iteration = max(iterations)
            else:
                iteration = int(args.iteration)
            model_path = f"{model_dir}{phone}_iteration_{iteration}.ckpt"
            # Only check for .index and .data-00000-of-00001 (not .meta)
            for ext in [".index", ".data-00000-of-00001"]:
                if not os.path.exists(model_path + ext):
                    print(f"Error: Model file missing: {model_path + ext}")
                    return
            print(f"Loading model: {model_path}")
            saver.restore(sess, model_path)

        print("Enhancing image...")
        enhanced_image = process_image_in_patches(sess, enhanced, x_, image, patch_size)
        enhanced_image_uint8 = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)

        # Save to enhanced_picture/ with incremental number
        save_folder = "enhanced_picture"
        output_path = get_next_incremental_filename(save_folder, ext=".png")
        try:
            imageio.imwrite(output_path, enhanced_image_uint8)
            print(f"Enhanced image saved to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            return

if __name__ == "__main__":
    main()
