import os
from glob import glob
from diffusers import StableDiffusionPipeline
import torch
import argparse


# 在原始数量图片上继续生成剩余所需图片
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Path to dreambooth file")
    parser.add_argument("--prompt", type=str, help="a photo of disa leaf")
    parser.add_argument("--output_directory", type=str, help="Path to the output directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")

    args = parser.parse_args()
    return args


args = parse_args()

pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")

categories = {
    "tomato_bacterial_spot_mild": "bacterial",
    "tomato_early_blight_mild": "early",
    "tomato_healthy": "healthy",
    "tomato_late_blight_mild": "late",
    "tomato_leaf_mold_mild": "mold",
    "tomato_mosaic_virus_mild": "mosaic",
    "tomato_septoria_leaf_spot_mild": "spot",
    "tomato_spider_mites_mild": "mite",
    "tomato_target_spot_mild": "target",
    "tomato_yellow_leaf_curl_virus_mild": "yellow",
    "grape_black_measles_mild": "measles",
    "grape_black_rot_mild": "rot",
    "grape_healthy": "healthy",
    "grape_leaf_blight_mild": "blight",
    "apple_black_rot_mild": "rot",
    "apple_healthy": "healthy",
    "apple_rust_mild": "rust",
    "apple_scab_mild": "scab"
}

prompt = args.prompt
class_name = prompt.split(' ')[3]
prompt = prompt.replace(class_name, categories.get(class_name))

img_name_p = prompt.replace(" ", "_")

# 创建结果图片文件夹
directory = os.path.join(args.output_directory, class_name)
if not os.path.exists(directory):
    os.mkdir(directory)

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
