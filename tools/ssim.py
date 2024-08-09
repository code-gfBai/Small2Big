import shutil
import argparse
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--classname', type=str)
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    return args


def compare_images(image1_path, image2_path):
    # 打开图像并转换为灰度图
    image1 = Image.open(image1_path).convert('L').resize((224, 224))
    image2 = Image.open(image2_path).convert('L').resize((224, 224))

    # 将图像转换为NumPy数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 计算SSIM值
    similarity = ssim(image1_array, image2_array)

    return similarity


if __name__ == "__main__":
    args = parse_args()
    # step_list = [100, 150, 200, 250, 300]
    step_list = [100]
    for step in step_list:
        path1_gen = f"path with generated images"
        path2_ori = f"path with corresponding training images"
        class_list = os.listdir(path1_gen)
        for classname in class_list:
            image1_path = f"{path1_gen}/{classname}/"
            image2_path = f"{path2_ori}/{classname}/"
            output_path = f"path/{classname.replace('_', ' ')}/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            img1_list = os.listdir(image1_path)
            img2_list = os.listdir(image2_path)

            data = {'train': [], 'test': [], 'SSIM': []}
            df = pd.DataFrame(data)
            for frequency in range(1):
                for test_name in tqdm(img2_list):
                    max_ssim = -2.0
                    # max_ssim = 10000
                    max_gen_path = None
                    test_path = os.path.join(image2_path, test_name)
                    for generated_name in img1_list:
                        generated_path = os.path.join(image1_path, generated_name)
                        similarity = compare_images(test_path, generated_path)
                        if similarity > max_ssim:
                            max_ssim = similarity
                            max_gen_path = generated_name
                    if max_gen_path is not None:
                        generated_name = max_gen_path
                        df = df.append({'train': test_name, 'test': generated_name, 'SSIM': max_ssim}, ignore_index=True)
                        if not os.path.exists(os.path.join(output_path, generated_name)):
                            shutil.copy2(os.path.join(image1_path, generated_name), os.path.join(output_path, generated_name))
                            img1_list.remove(generated_name)
                        print("最高的SSIM值为：", max_ssim)
            # 将 DataFrame 存入 Excel 文件
            # excel_file = f'/home/swu/bgf/ResNet/image/excel/{classname}_step_{step}_ssim_results.xlsx'
            excel_path = "path to excel with ssim"
            if not os.path.exists(excel_path):
                os.makedirs(excel_path)
            excel_file = f'{excel_path}/{classname}_ssim_results.xlsx'
            df.to_excel(excel_file, index=False)
            print(f'SSIM values saved to {excel_file}')

