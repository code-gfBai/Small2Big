import os
from glob import glob
from diffusers import StableDiffusionPipeline
import torch
import argparse

#在原始数量图片上继续生成剩余所需图片
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Path to dreambooth file")
    # parser.add_argument("--inversion_path", type=str, help="Path to the inversion file")
    # parser.add_argument("--token", type=str, help="the token in text prompt")
    parser.add_argument("--prompt", type=str, help="a photo of disa leaf")
    parser.add_argument("--output_directory", type=str, help="Path to the output directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")

    args = parser.parse_args()
    return args

args = parse_args()

pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")

prompt = args.prompt

img_name_p = prompt.replace(" ", "_")

# 创建结果图片文件夹
directory = args.output_directory
if not os.path.exists(directory):
    os.mkdir(args.output_directory)

# 计算已有图片数量
existing_images = glob(os.path.join(directory, "*.png"))
num_existing_images = len(existing_images)
print("existing_images="+str(num_existing_images))
# 计算还需要生成多少张图片
remaining_images = args.num_images - num_existing_images

output_category = os.path.basename(args.output_directory)
# 生成剩余的图片
inference_steps_values = [50, 100, 150, 200]
# inference_steps_values = [100, 100, 100, 100]
for i in range(remaining_images):
    image = pipe(prompt.replace("_", " "), num_inference_steps=inference_steps_values[i % 4], guidance_scale=7.5).images[0]
    # image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
    # image = image.resize((256, 256))
    image.save(os.path.join(directory, f"{output_category}_{img_name_p}_inference_steps_{inference_steps_values[i % 4]}_{num_existing_images + i}.png"))
