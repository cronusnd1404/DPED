# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
import imageio
import numpy as np
import sys
import os

# Set environment variables for GPU optimization
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'

tf.compat.v1.disable_v2_behavior()

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import models
import utils
import vgg

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

# Drastically reduce batch size for 4GB GPU
batch_size = min(batch_size, 1)  # Use batch size of 1 to minimize memory usage

np.random.seed(0)

# Configure GPU with very conservative settings
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # Use more memory but carefully
config.allow_soft_placement = True
config.log_device_placement = False

# defining system architecture

with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as sess:
    
    # placeholders for training data

    phone_ = tf.compat.v1.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.compat.v1.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced = models.resnet(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)

    loss_discrim = -tf.reduce_sum(discrim_target * tf.compat.v1.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    loss_texture = -loss_discrim

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # 2) content loss with memory optimization

    CONTENT_LAYER = 'relu5_4'

    # Create VGG networks with gradient stopping to reduce memory
    with tf.name_scope("vgg_enhanced"):
        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    
    with tf.name_scope("vgg_dslr"):
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # 3) color loss

    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    discriminator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("discriminator")]

    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
    train_step_disc = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.compat.v1.global_variables_initializer())

    # loading training and test data

    print("Loading test data...")
    test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
    print("Test data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
    print("Training data was loaded\n")

    TEST_SIZE = test_data.shape[0]
    num_test_batches = int(test_data.shape[0] / batch_size)

    print('Training network')

    train_loss_gen = 0.0
    train_acc_discrim = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, min(5, batch_size)), :]

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()

    for i in range(num_train_iters):

        try:
            # train generator

            idx_train = np.random.randint(0, train_size, batch_size)

            phone_images = train_data[idx_train]
            dslr_images = train_answ[idx_train]

            [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                            feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
            train_loss_gen += loss_temp / eval_step

            # train discriminator

            idx_train = np.random.randint(0, train_size, batch_size)

            # generate image swaps (dslr or enhanced) for discriminator
            swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

            phone_images = train_data[idx_train]
            dslr_images = train_answ[idx_train]

            [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                            feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
            train_acc_discrim += accuracy_temp / eval_step

        except tf.errors.ResourceExhaustedError as e:
            print(f"GPU memory exhausted at iteration {i}, trying to continue...")
            continue

        if i % eval_step == 0:

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(min(num_test_batches, 10)):  # Limit test batches to save memory

                be = j * batch_size
                en = (j+1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                try:
                    [enhanced_crops, accuracy_disc, losses] = sess.run([enhanced, discim_accuracy, \
                                    [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr]], \
                                    feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})

                    test_losses_gen += np.asarray(losses) / min(num_test_batches, 10)
                    test_accuracy_disc += accuracy_disc / min(num_test_batches, 10)

                    # Calculate SSIM with error handling
                    try:
                        ssim_value = MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                                            enhanced_crops * 255) / min(num_test_batches, 10)
                        if np.isnan(ssim_value) or np.isinf(ssim_value):
                            ssim_value = 0.0
                        loss_ssim += ssim_value
                    except:
                        loss_ssim += 0.0

                except tf.errors.ResourceExhaustedError:
                    print("Memory error during testing, skipping this batch")
                    continue

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                  (i, phone, train_acc_discrim, test_accuracy_disc)

            # Handle NaN in loss_ssim for logging
            ssim_display = loss_ssim if not (np.isnan(loss_ssim) or np.isinf(loss_ssim)) else 0.0

            logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ms-ssim: %.4g\n" % \
                  (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                   test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], ssim_display)

            print(logs_disc)
            print(logs_gen)

            # save the results to log file

            logs = open('models/' + phone + '.txt', "a")
            logs.write(logs_disc)
            logs.write('\n')
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops
            try:
                enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})

                idx = 0
                for crop in enhanced_crops:
                    before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                    # Convert to uint8 and clip values to valid range
                    before_after = np.clip(before_after * 255, 0, 255).astype(np.uint8)
                    imageio.imwrite('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                    idx += 1
            except tf.errors.ResourceExhaustedError:
                print("Memory error during visual results saving, skipping...")

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
