from sklearn.cluster import KMeans
import shutil
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class net(nn.Module):
    def __init__(self):

        super(net, self).__init__()
        self.net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output

# 图像预处理和特征提取函数
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(images):
    batch = torch.stack([preprocess(image) for image in images])
    batch = Variable(batch.float().cuda(), requires_grad=False)
    with torch.no_grad():
        model = net().cuda()
        batch_features = model(batch)
        batch_features = torch.squeeze(batch_features)
        batch_features = batch_features.detach().cpu().numpy()
    return batch_features

def clusters_select(class_name, path, n_clusters):
    # 加载图像文件夹中的所有图像
    image_folder = f"{path}/{class_name}"  # 图片文件夹路径
    image_files = os.listdir(image_folder)

    # 定义批处理大小和图像总数
    batch_size = 256
    num_images = len(image_files)

    # 读取图像并提取特征
    images = []
    features = []
    for i in tqdm(range(0, num_images, batch_size)):
        batch_images = []
        for j in range(i, min(i + batch_size, num_images)):
            image_file = image_files[j]
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            batch_images.append(image)

        batch_features = extract_features(batch_images)  # Extract features for the batch

        # Save the features for each image in the batch
        for k in range(len(batch_images)):
            image_file = image_files[i + k]
            feature = batch_features[k]
            images.append(batch_images[k])
            features.append(feature)
    # 转换为NumPy数组
    features = np.array(features)

    # 执行层次聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    labels = kmeans.labels_

    # 计算每个类别的中心特征向量
    centroid_features = kmeans.cluster_centers_

    # 创建类别文件夹
    output_folder = f"{path}/output_kmeans/{class_name}"
    os.makedirs(output_folder, exist_ok=True)

    # 选择每个类别中与中心最接近的图像复制为中心图像
    for cluster_label, centroid_feature in enumerate(centroid_features):
        distance_list = []
        cluster_indices = np.where(labels == cluster_label)[0]  # 获取属于当前类别的索引列表
        for index in cluster_indices:
            feature = features[index]
            distance = np.linalg.norm(feature - centroid_feature)  # 计算特征向量之间的欧氏距离
            distance_list.append(distance)

        closest_index = cluster_indices[np.argmin(distance_list)]  # 找到与中心最接近的图像索引
        closest_image = images[closest_index]
        cluster_folder = os.path.join(output_folder, f"Cluster_{cluster_label}")

        closest_image_path = os.path.join(cluster_folder, f"Centroid_{cluster_label}.jpg")
        # shutil.copy(f'{image_folder}/{image_files[closest_index]}', f'{output_folder}/Centroid_{cluster_label}.jpg')
        shutil.copy(f'{image_folder}/{image_files[closest_index]}', f'{output_folder}/{image_files[closest_index]}')
        print(f"Centroid Image for Cluster {cluster_label} saved at: {output_folder}/{image_files[closest_index]}")


def has_target_words(string):   # 选取物种
    target_words = ['apple', 'grape', 'tomato']
    for word in target_words:
        if word in string:
            return True
    return False


path = r"path to image" # 图片文件夹
class_list = os.listdir(path)
#
for class_name in class_list:
    if 'healthy' in class_name.lower():  # 如果文件夹名包含 "healthy"
        clusters_select(class_name, path, 200)  # 这里规定聚类的数量
    else: # 如果是病害类
        clusters_select(class_name, path, 20)  # 这样就可以同时健康，病害聚类不同的图片数量

