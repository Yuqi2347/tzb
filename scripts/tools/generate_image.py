# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import argparse

def generate_images(output_dir, resolutions, num_per_res):
    """
    生成随机 RGB 图像

    Args:
        output_dir (str): 输出文件夹
        resolutions (list of tuple): 每个分辨率 (H, W)
        num_per_res (int): 每个分辨率生成的图像数量
    """
    os.makedirs(output_dir, exist_ok=True)

    for res in resolutions:
        h, w = res
        res_dir = os.path.join(output_dir, f"{h}x{w}")
        os.makedirs(res_dir, exist_ok=True)

        for i in range(num_per_res):
            # 随机生成图像
            img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_file = os.path.join(res_dir, f"img_{i+1:03d}.png")
            img.save(img_file)

        print(f"生成 {num_per_res} 张 {h}x{w} 图像到 {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成随机图像")
    parser.add_argument("--output-dir", type=str, default="./generated_images",
                        help="输出文件夹")
    parser.add_argument("--resolutions", type=str, default="224x224,384x384,512x512",
                        help="分辨率列表，用逗号分隔，例如 224x224,384x384")
    parser.add_argument("--num-per-res", type=int, default=10,
                        help="每个分辨率生成的图像数量")
    args = parser.parse_args()

    # 解析分辨率字符串
    res_list = []
    for r in args.resolutions.split(","):
        h, w = map(int, r.lower().split("x"))
        res_list.append((h, w))

    generate_images(args.output_dir, res_list, args.num_per_res)
