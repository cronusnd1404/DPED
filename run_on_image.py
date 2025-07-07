import os
import sys
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf

from models import generator
from utils import preprocess_test_image, save_image

# Đường dẫn mặc định
DEFAULT_MODEL_CKPT = 'models/iphone_iteration_20000.ckpt'
DEFAULT_OUTPUT_DIR = 'results/'


def load_generator_model(ckpt_path):
    model = generator()
    checkpoint = tf.train.Checkpoint(generator=model)
    checkpoint.restore(ckpt_path).expect_partial()
    return model


def run_model_on_image(input_path, output_dir, ckpt_path):
    # Đọc ảnh
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)
    img_pre = preprocess_test_image(img_np)
    img_pre = np.expand_dims(img_pre, axis=0)

    # Nạp model
    model = load_generator_model(ckpt_path)

    # Chạy model
    output = model(img_pre, training=False)
    output_img = np.clip(output[0].numpy(), 0, 255).astype(np.uint8)

    # Lưu ảnh kết quả
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_path))
    save_image(output_img, out_path)
    print(f"Saved result to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Run model on your image or folder of images')
    parser.add_argument('--input', required=True, help='Path to your input image or folder')
    parser.add_argument('--ckpt', default=DEFAULT_MODEL_CKPT, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save result')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # Nếu là thư mục, xử lý tất cả ảnh trong thư mục
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(exts)]
        if not files:
            print(f"No images found in {args.input}")
            return
        for f in files:
            print(f"Processing {f}...")
            run_model_on_image(f, args.output_dir, args.ckpt)
    elif os.path.isfile(args.input):
        run_model_on_image(args.input, args.output_dir, args.ckpt)
    else:
        print(f"Input path {args.input} does not exist.")
        return

if __name__ == '__main__':
    main()
