import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file_path = r"path to images".replace('\\', '/')
plant_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

folders = ['train', 'val', 'test']

file_path_1 = r"path to output of data count".replace('\\', '/')
for folder in folders:
    folder_path = os.path.join(file_path_1, folder)
    mkfile(folder_path)
    for cla in plant_class:
        mkfile(os.path.join(folder_path, cla))

split_rate_train = 0.6
split_rate_val = 0.2
split_rate_test = 0.2


for cla in plant_class:
    cla_path = os.path.join(file_path, cla)
    images = os.listdir(cla_path)
    num = len(images)

    train_index = random.sample(images, k=int(num * split_rate_train))
    remaining_images = set(images) - set(train_index)

    val_test_size = int(num * (split_rate_val + split_rate_test))
    val_test_index = random.sample(remaining_images, k=val_test_size)

    val_size = int(val_test_size * split_rate_val / (split_rate_val + split_rate_test))
    val_index = val_test_index[:val_size]
    test_index = val_test_index[val_size:]

    for index, image in enumerate(images):
        image_path = os.path.join(cla_path, image)
        if image in train_index:
            new_path = os.path.join(file_path_1, 'train', cla, image)
        elif image in val_index:
            new_path = os.path.join(file_path_1, 'val', cla, image)
        else:
            new_path = os.path.join(file_path_1, 'test', cla, image)
        copy(image_path, new_path)
        print("\r[{}] 处理中 [{}/{}]".format(cla, index + 1, num), end="")
    print()

print("处理完成！")
