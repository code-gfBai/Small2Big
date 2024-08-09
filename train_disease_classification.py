import time
import os
import logging
import argparse

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import shutil

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

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
        accuracy.append(rec)

        # 计算F1分数
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (prec * rec) / (prec + rec)
        f1_score.append(f1)

    return accuracy, precision, recall, f1_score


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

    class_to_idx = image_datasets['val'].class_to_idx
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=True),
    }
    return dataloaders, class_to_idx


# 模型训练函数
def train_model(model, dataloader, criterion, optimizer, device, logger, class_num):
    model.train()
    running_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=True)  # 创建进度条
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        #清除历史梯度
        optimizer.zero_grad()

        outputs = model(inputs)
        pred = torch.nn.Softmax(dim=1)(outputs)
        # pred = torch.round(pred * 1000) / 1000
        _, preds = torch.max(pred, 1)
        # _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels) #计算损失值

        loss.backward()#反向传播
        optimizer.step()#更新优化器参数

        running_loss += loss.item()
        # corrects += torch.sum(preds == labels.data)

        # 更新进度条的显示信息
        progress_bar.set_postfix({"Loss": loss.item()})

        labels_list = labels.data.tolist()
        preds_list = preds.tolist()

        all_preds.extend(preds_list)
        all_labels.extend(labels_list)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy, precision, recall, F1_score = compute_metrics(conf_matrix, class_num)
    # overall_accuracy = np.mean(accuracy)
    overall_accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    epoch_loss = running_loss / len(dataloader)
    # epoch_acc = corrects.double() / len(dataloader.dataset)

    # 记录日志
    logger.info(f"Training Loss: {epoch_loss:.4f}, Accuracy: {overall_accuracy:.4f}")
    return epoch_loss, overall_accuracy

def eval_model(model, dataloader, criterion, device, logger, class_to_idx, class_num):
    model.eval()
    running_loss = 0.0
    corrects = 0
    progress_bar = tqdm(dataloader, desc="Evaluation", leave=True)

    img_num = [0] * class_num
    pred_num = [0] * class_num
    class_acc = [0.] * class_num
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # print("input="+inputs+" labels="+labels)
            outputs = model(inputs)

            pred = torch.nn.Softmax(dim=1)(outputs)
            # pred = torch.round(pred * 1000) / 1000
            _, preds = torch.max(pred, 1)

            # _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            # corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix({"Loss": loss.item()})
            labels_list = labels.data.tolist()
            preds_list = preds.tolist()
            # for i in range(len(labels_list)):
            #     img_num[labels_list[i]] += 1
            #     if labels_list[i] == preds_list[i]:
            #         pred_num[labels_list[i]] += 1
            all_preds.extend(preds_list)
            all_labels.extend(labels_list)
        # logger.info(f"img_num: {img_num}")
        # logger.info(f"pred_num: {pred_num}")
        # for i in range(len(img_num)):
        #     class_acc[i] = pred_num[i] / img_num[i]
        #     logger.info(f"{find_key(class_to_idx, i)}: {class_acc[i]}")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy, precision, recall, F1_score = compute_metrics(conf_matrix, class_num)
    # overall_accuracy = np.mean(accuracy)
    overall_accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    epoch_loss = running_loss / len(dataloader)
    # epoch_acc = corrects.double() / len(dataloader.dataset)

    # epoch_acc = np.mean(class_acc)  # 计算总体平均准确率

    # Log the results
    logger.info(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {overall_accuracy:.4f}")
    return epoch_loss, overall_accuracy


def test_model(model, dataloader, criterion, device, logger, class_to_idx, class_num):
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

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred = torch.nn.Softmax(dim=1)(outputs)
            # pred = torch.round(pred * 1000) / 1000
            _, preds = torch.max(pred, 1)

            # _, preds = torch.max(outputs, 1)

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
        accuracy, precision, recall, F1_score = compute_metrics(conf_matrix, class_num)
        # f"Accuracy: {str(format(accuracy[i] * 100, '.2f')).rjust(6, ' ')}, "
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
    # epoch_acc = corrects.double() / len(dataloader.dataset)

    # epoch_acc = np.mean(class_acc)  # 计算总体平均准确率
    # Log the results
    logger.info(f"Test Loss: {epoch_loss:.4f}, Accuracy: {overall_accuracy:.4f}")
    return epoch_loss, overall_accuracy

def main():
    # 解析参数
    args = parse_args()

    # 准备数据
    dataloaders,class_to_idx = prepare_data(args)
    # print("args_num:"+args.class_num)
    # from IPython import embed
    # embed()
    # 判断是否加载预训练模型还是自定义权重
    if args.weights_loc is None:
        # # 加载预训练的ResNet-50模型
        model = models.resnet50(pretrained=True)
        # model = models.resnet34(pretrained=True)
        # model = models.resnet18(pretrained=True)

        # 修改全连接层，使其适应你的分类任务
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=args.class_num)

    else:
        # 加载已训练好的权重文件
        model = torch.load(args.weights_loc)
    # 加载预训练的ResNet-50模型
    # model = models.resnet50(pretrained=True)
    # model.fc = torch.nn.Linear(2048, args.class_num)
    # 如果有GPU可用，将模型移到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=0.000001, nesterov=True)
    # 获取当前时间
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    #定义模型保存的文件夹路径并创建
    model_folder = os.path.join(args.output_path,f"{args.output_folder_name}_{args.net_name}_weight_{current_time}_epochs{args.epochs}")
    os.makedirs(model_folder, exist_ok=True)

    #log
    log_folder = os.path.join(model_folder,'logs')
    os.makedirs(log_folder,exist_ok=True)
    log_path = os.path.join(log_folder,'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个SummaryWriter，指定日志目录
    writer_path = os.path.join(log_folder,'writer_show_res')
    os.makedirs(writer_path,exist_ok=True)
    writer = SummaryWriter(writer_path)

    # 创建logger对象
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    logger.info('Starting training...')

    best_acc = 0.0  # 初始化最佳准确率为0
    best_model_wts = model.state_dict()
    # since = time.time()
    epoch_times = []  # 用于保存每个 epoch 的训练时间
    # total_epoch = args.epochs+args.resume_epoch-1
    for epoch in range(args.resume_epoch,args.epochs+1):
        logger.info(f'Epoch {epoch }/{args.epochs}')
        logger.info('-' * 20)

        start_time = time.time()  # 记录当前 epoch 的开始时间

        train_loss, train_acc = train_model(model, dataloaders['train'], criterion, optimizer, device, logger, args.class_num)
        # val_loss, val_acc = eval_model(model, dataloaders['val'], criterion, device,logger)
        val_loss, val_acc = eval_model(model, dataloaders['val'], criterion, device, logger, class_to_idx, args.class_num)
        # test_loss, test_acc = test_model(model, dataloaders['test'], criterion, device, logger, class_to_idx, args.class_num)

        # 记录当前 epoch 的训练时间以及平均时间
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / epoch

        logger.info(f'Epoch {epoch} needs time: {epoch_time:.0f}s')
        logger.info(f'Total time spent to the current {epoch} epoch:{sum(epoch_times) :.0f}s')
        logger.info(f'Average epoch time for the {epoch} epochs: {avg_epoch_time:.0f}s')

        # 记录损失函数
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        # writer.add_scalar('Loss/test', test_loss, epoch)
        # 记录准确率
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)
        # 关闭SummaryWriter
        writer.close()

        logger.info(f'Train - Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        logger.info(f'Val - Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        # logger.info(f'Test - Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

        # 保存每10个epoch的模型权重
        if epoch % 10 == 0:
            save_path = os.path.join(model_folder, f'{args.net_name}_epoch_{epoch}.pth')
            torch.save(model, save_path)

        # 保存模型并更新最佳准确率
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            model.load_state_dict(best_model_wts)
            torch.save(model, os.path.join(model_folder, f'{args.net_name}_best_epoch{epoch}.pth'))
            logger.info(f"Model saved at {os.path.join(model_folder, f'{args.net_name}_best_epoch{epoch}.pth')}")
    # # 保存最终模型
    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(model_folder, f'{args.net_name}_.pth'))
    logger.info(f"Model saved at {os.path.join(model_folder, f'{args.net_name}_.pth')}")

    # 计算并记录平均 epoch 时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    logger.info(f'Total {len(epoch_times)} epochs average epoch time: {avg_epoch_time:.0f}s')

    logger.info('Start Test:')
    test_loss, test_acc = test_model(model, dataloaders['test'], criterion, device, logger, class_to_idx,
                                     args.class_num)
    logger.info(f'Test - Loss: {test_loss:.4f} Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()
