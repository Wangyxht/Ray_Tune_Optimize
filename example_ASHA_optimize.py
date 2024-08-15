import os
from functools import partial

import torch
from ray.train import Checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim, cuda
from MinstNet import MinstNet
from ray import train, tune
from optimize import optimize_random_search


def load_data(data_dir="./data"):
    """
    从数据集文件夹中获取数据
    :param data_dir: 数据集文件夹名
    :return: 数据集
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    return train_set, test_set


def train_minst(config, data_dir=None):
    """
    训练MINST神经网络模型，并且报告loss与验证集上的精确度
    :param config: 调优参数以及待传递参数
    """
    # 实例化模型
    minst_net = MinstNet(l1=config['l1'], l2=config['l2'])

    # 适配训练设备，测试节点是否为单机多卡
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            minst_net = nn.DataParallel(minst_net)
    minst_net.to(device)

    # 获取数据集
    train_set, test_set = load_data(data_dir=data_dir)
    train_dataloader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=config['batch_size'])

    # 定义损失函数与优化器
    lossF = nn.CrossEntropyLoss()  # 损失函数为交叉熵
    optimizer = optim.SGD(minst_net.parameters(), lr=config['lr'])  # 优化器

    while True:
        # 训练模型
        total_loss = 0
        minst_net.train(True)
        for _, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            minst_net.zero_grad()
            outputs = minst_net(images)
            loss = lossF(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 验证模型
        minst_net.train(False)
        total_accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = minst_net(images)
                loss = lossF(outputs, labels)
                test_loss += loss.item()
                accuracy = (outputs.argmax(1) == labels).sum().item()
                total_accuracy = total_accuracy + accuracy

        # 报告当前训练的损失函数值以及测试集准确率
        # 注意在此处是每训练一个epoch报告一次，以便调度器调度
        metrics = {
            "test_accuracy": total_accuracy / len(test_set),
            "train_loss": total_loss / len(train_dataloader),
            "test_loss": test_loss / len(test_dataloader),
        }
        train.report(metrics)
    # end while True

if __name__ == '__main__':
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    # 定义超参数空间
    config = {
        "l1": tune.choice([512, 256]),  # 第一隐藏层节点数
        "l2": tune.choice([128, 64, 32, 16]),  # 第二隐藏层节点数
        "lr": tune.loguniform(0.01, 0.1),  # 学习率
        "batch_size": tune.choice([128, 64, 32, 16]),  # 批次大小
    }

    # 引入ASHA算法的随机搜索
    results, best_result = optimize_random_search(
        objective=partial(train_minst, data_dir=data_dir),  # 目标函数，partial为传递除config以外的其它参数
        config=config,  # 目标超参数搜索空间
        metric="test_accuracy",  # 优化目标变量
        mode="max",  # 优化模式，可选max/min
        n_samples=20,  # 采样点个数
        max_iter=10,  # 最大迭代次数
        scheduler="ASHA",  # 引入的算法调度器，实现早期不良训练的淘汰
        grace_period=2,  # 最小训练轮数，起始2轮后开始淘汰，2轮前ASHA不介入
        reduce_factor=2,  # 淘汰率,即论文/伪代码中的eta
        cpus_per_trial=1,  # 每个实验分配的cpu资源
        gpus_per_trial=0.2,  # 每个实验分配的gpu资源
    )
