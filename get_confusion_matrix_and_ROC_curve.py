import time
import os
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
import shutil

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns






def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def compute_metrics(cm, class_num):

    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for class_id in range(class_num):

        tp = cm[class_id, class_id]  # 对角线元素
        fp = cm[:, class_id].sum() - tp  # 该列除对角线元素外其他元素之和
        fn = cm[class_id, :].sum() - tp  # 该行除对角线元素外其他元素之和
        tn = cm.sum() - (tp + fp + fn)

        # 计算准确率
        # tn = 0
        total = tp + tn + fp + fn
        acc = (tp + tn) / total

        accuracy.append(acc)

        # 计算精确率
        if tp + fp == 0:
            prec = 0.0
        else:
            prec = tp / (tp + fp)
        precision.append(prec)

        # 计算召回率
        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)
        recall.append(rec)

        # 计算F1分数
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (prec * rec) / (prec + rec)
        f1_score.append(f1)

    return accuracy, precision, recall, f1_score

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (per device) for the training dataloader.')
    # parser.add_argument('--data_dir', type=str, default='/home/swu/cjh/ResNet/data', help='A folder containing the training data.')
    parser.add_argument('--train_data_dir', type=str, default='/path/to/train', help='A folder containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='/home/swu/cjh/ResNet/data/val', help='A folder containing the validation data.')
    parser.add_argument('--test_data_dir', type=str, default='/home/swu/cjh/ResNet/data/test', help='A folder containing the test data.')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--output_path', type=str, default='resnet50.pth', help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--output_folder_name', type=str, default='o_num4_resnet50',help='The output res folder name.')
    parser.add_argument('--net_name', type=str, default='resnet50', help='The output res folder name.')
    # parser.add_argument('--excel_path', type=str, default='resnet50_val_acc.xlsx',
    #                     help='The output val_acc excel and chart path ')
    # parser.add_argument('--log_path', type=str, default='log', help='log directory. Will default to')
    parser.add_argument('--class_num', type=int, default='38', help='the class number we need train')
    parser.add_argument('--lr',type=float,default=0.01,help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument('--weights_loc', type=str, default=None, help='path of weights (if going to be loaded)')
    parser.add_argument('--resume_epoch', type=int, default=1, help='what epoch to start from')
    args = parser.parse_args()
    return args

# 数据预处理和加载器定义
def prepare_data(args):#加载数据 返回test文件夹中文件名称
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(args.train_data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(args.val_data_dir, data_transforms['val']),
        'test': datasets.ImageFolder(args.test_data_dir, data_transforms['test']),
    }

    class_to_idx = image_datasets['test'].class_to_idx
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=True),
    }
    return dataloaders, class_to_idx


def test_model(model, dataloader, criterion, device, logger, class_to_idx, class_num, matrix_path):
    model.eval()
    running_loss = 0.0
    corrects = 0
    progress_bar = tqdm(dataloader, desc="Test", leave=True)

    img_num = [0] * class_num
    pred_num = [0] * class_num
    class_acc = [0.] * class_num

    all_preds = []
    all_labels = []
    # misclassified_data = {}  # Dictionary to store misclassified image data by class
    all_probs = []  # 添加这行来存储预测概率

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred = torch.nn.Softmax(dim=1)(outputs)
            #保存预测代码
            all_probs.extend(pred.cpu().numpy())

            _, preds = torch.max(pred, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            # corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix({"Loss": loss.item()})

            labels_list = labels.data.tolist()
            preds_list = preds.tolist()
            for i in range(len(labels_list)):
                img_num[labels_list[i]] += 1
                if labels_list[i] == preds_list[i]:
                    pred_num[labels_list[i]] += 1

            all_preds.extend(preds_list)
            all_labels.extend(labels_list)

        logger.info(f"img_num: {img_num}")
        logger.info(f"pred_num: {pred_num}")


        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(classification_report(all_labels, all_preds))
        # precision, recall, F1_score = compute_metrics(np.array(all_preds), np.array(all_labels), class_num)
        accuracy, precision, recall, F1_score = compute_metrics(conf_matrix, class_num)

        for i in range(class_num):
            logger.info(
                f"{find_key(class_to_idx, i).ljust(40, ' ')} - "
                f"Precision: {str(format(precision[i]*100,'.2f')).rjust(6, ' ')}, "
                f"Recall: {str(format(recall[i]*100,'.2f')).rjust(6, ' ')}, "
                f"F1 Score: {str(format(F1_score[i]*100,'.2f')).rjust(6, ' ')}")
        # overall_accuracy = np.mean(accuracy)
        overall_accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
        overall_precision = np.mean(precision)
        overall_recall = np.mean(recall)
        overall_f1_score = np.mean(F1_score)
        # f"Accuracy: {str(format(overall_accuracy * 100, '.2f')).rjust(6, ' ')}, "
        logger.info(f"{'Overall'.ljust(40, ' ')} - "
                    f"Precision: {str(format(overall_precision*100,'.2f')).rjust(6, ' ')}, "
                    f"Recall: {str(format(overall_recall*100,'.2f')).rjust(6, ' ')}, "
                    f"F1 Score: {str(format(overall_f1_score*100,'.2f')).rjust(6, ' ')}")

    epoch_loss = running_loss / len(dataloader)

    # epoch_acc = np.mean(class_acc)  # 计算总体平均准确率
    # Log the results
    logger.info(f"Test Loss: {epoch_loss:.4f}, Accuracy: {format(overall_accuracy*100,'.2f')}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = [find_key(class_to_idx, i) for i in range(class_num)]

    # 打印混淆矩阵

    max_label_length = max([len(name) for name in class_names])
    figsize = max(len(class_names), max_label_length * 0.5)
    #
    # plt.figure(figsize=(figsize, figsize))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    # 创建颜色映射数组
    colors_array = np.full(conf_matrix.shape, 'lightyellow', dtype='object')  # 全部初始化为淡黄色

    # 主对角线上根据数值大小设置蓝色深浅
    diagonal_values = np.diag(conf_matrix)
    max_val = diagonal_values.max()
    min_val = diagonal_values.min()
    norm = plt.Normalize(min_val, max_val)  # 归一化
    for i in range(len(class_names)):
        color_intensity = norm(diagonal_values[i])
        colors_array[i, i] = plt.cm.Blues(color_intensity)

    # 绘制混淆矩阵
    plt.figure(figsize=(figsize, figsize))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', mask=colors_array == 'lightyellow',
    #             xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 20})
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=['lightyellow'], mask=colors_array != 'lightyellow',
    #             xticklabels=class_names, yticklabels=class_names, cbar=False, annot_kws={"fontsize": 20})
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 20})

    plt.xlabel('Predicted', fontsize=25)
    plt.ylabel('True', fontsize=25)
    plt.title('Confusion Matrix', fontsize=25)

    # 使用tight_layout确保图表元素不会重叠
    plt.tight_layout()

    # 保存混淆矩阵图到指定路径
    plt.savefig(os.path.join(matrix_path, "confusion_matrix.png"))
    plt.close()

    # 绘制并保存每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_labels_array = np.array(all_labels)
    all_preds_prob = np.array(all_probs)


    # 计算每个类别的FPR, TPR和AUC
    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(all_labels_array == i, all_preds_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 设置颜色循环
    colors = cycle(['royalblue', 'indianred', 'brown', 'pink','palegreen', 'teal','khaki', 'darkmagenta', 'tan', 'orange', 'darkgreen', 'cyan', 'plum', 'forestgreen',  'gray', 'olive',  'gold', ])
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # 设置y轴范围从0.9到1.0
    plt.ylim([0.9, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver Operating Characteristic for multi-class', fontsize=15)
    plt.legend(loc="lower right")

    # 保存ROC曲线图到指定路径
    plt.savefig(os.path.join(matrix_path, "roc_curves.png"))
    plt.close()


    return epoch_loss, overall_accuracy


def main():
    # 解析参数
    args = parse_args()
    since = time.time()
    # 准备数据
    dataloaders, class_to_idx = prepare_data(args)
    # print("args_num:"+args.class_num)

    # 判断是否加载预训练模型还是自定义权重
    if args.weights_loc is None:
        # 加载预训练的ResNet-50模型
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, args.class_num)

    else:
        # 加载已训练好的权重文件
        model = torch.load(args.weights_loc)

    # 加载预训练的ResNet-50模型
    # model = models.resnet50(pretrained=True)
    # model.fc = torch.nn.Linear(2048, args.class_num)

    # 如果有GPU可用，将模型移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.cuda()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 获取当前时间
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    #定义模型保存的文件夹路径并创建
    # model_folder = os.path.join(args.output_path,f"{args.output_folder_name}_weight_{current_time}")
    # os.makedirs(model_folder, exist_ok=True)

    #log
    log_folder = os.path.join(args.output_path,'test_logs_conf_matrix')
    # log_folder=args.text_log_folder_name #测试结果的log文件写入对应模型的log文件夹中
    os.makedirs(log_folder,exist_ok=True)

    log_path = os.path.join(log_folder,f'text_log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    # 创建logger对象
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    logger.info('Starting testing...')

    # since = time.time()
    epoch_times = []  # 用于保存每个 epoch 的训练时间

    # misclassified_folder = os.path.join(args.output_path, 'misclassified_images')
    # os.makedirs(misclassified_folder, exist_ok=True)
    #create matrix path
    matrix_path = log_folder
    os.makedirs(matrix_path,exist_ok=True)

    test_loss, test_acc = test_model(model, dataloaders['test'], criterion, device, logger, class_to_idx,
                                     args.class_num,  matrix_path)


if __name__ == '__main__':
    main()
