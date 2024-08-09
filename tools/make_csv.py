import os
import csv

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
    "tomato_yellow_leaf_curl_virus_mild": "yellow"
}
# categories = {
#     "grape_black_measles_mild": "measles",
#     "grape_black_rot_mild": "rot",
#     "grape_healthy": "healthy",
#     "grape_leaf_blight_mild": "blight"
# }
# categories = {
#     "apple_black_rot_mild": "rot",
#     "apple_healthy": "healthy",
#     "apple_rust_mild": "rust",
#     "apple_scab_mild": "scab"
# }

# 设置文件夹路径
folder_path = r"path to image"

# 设置 CSV 文件名
csv_file = f'{folder_path}/metadata.csv'

# 自定义文本
custom_text = 'a photo of disease leaf'
#
class_text = "a photo of healthy leaf"

# 创建并写入 CSV 文件
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_name', 'text'])  # 写入标题行

    # 遍历文件夹中的所有文件
    for cat in os.listdir(folder_path):
        if cat.endswith('csv'):
            continue
        path1 = os.path.join(folder_path, cat)
        for cal in os.listdir(path1):
            path2 = os.path.join(path1, cal)
            if cal == "tomato":
                for filename in os.listdir(path2):
                    if filename.endswith(('.jpg', 'JPG')):  # 检查文件是否是图像
                        image_path = os.path.join(path2, filename)
                        writer.writerow([f'{cat}/{cal}/{filename}', class_text])
            else:
                for filename in os.listdir(path2):
                    if filename.endswith(('.jpg', 'JPG')):  # 检查文件是否是图像
                        image_path = os.path.join(path2, filename)
                        writer.writerow([f'{cat}/{cal}/{filename}', custom_text.replace("disease", categories.get(cal))])


print(f'CSV 文件已创建：{csv_file}')
