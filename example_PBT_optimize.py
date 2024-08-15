import os
import tempfile
from functools import partial

import torch
from ray.train import Checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim, cuda
from MinstNet import MinstNet
from ray import train, tune
from optimize import optimize_pbt_search


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


def train_minst_with_checkpoint(config, data_dir=None):
    """
    训练MINST神经网络模型，并且报告loss与验证集上的精确度
    :param config: 调优参数以及待传递参数
    """
    # 实例化模型
    minst_net = MinstNet(l1=512, l2=64)
    # 适配训练设备，测试节点是否为单机多卡
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            minst_net = nn.DataParallel(minst_net)
    minst_net.to(device)
    # 定义损失函数与优化器
    lossF = nn.CrossEntropyLoss()  # 损失函数为交叉熵
    optimizer = optim.SGD(minst_net.parameters(), lr=config['lr'], momentum=config['momentum'])  # 优化器

    # If `train.get_checkpoint()` is populated,
    # then we are resuming from a checkpoint.
    if train.get_checkpoint():
        with train.get_checkpoint().as_directory() as checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            minst_net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            for param_group in optimizer.param_groups:
                if "lr" in config:
                    param_group["lr"] = config["lr"]
                if "momentum" in config:
                    param_group["momentum"] = config["momentum"]

    # 获取数据集
    train_set, test_set = load_data(data_dir=data_dir)
    train_dataloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=64)

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

        # 将本轮的训练模型保存到检查点
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (minst_net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                metrics=metrics,
                checkpoint=checkpoint,
            )
    # end while True


if __name__ == '__main__':
    data_dir = os.path.abspath("./data")
    load_data(data_dir)

    config = {
        "lr": tune.uniform(0.001, 0.1),
        "momentum": tune.uniform(0.1, 0.5),
    }

    results = optimize_pbt_search(
        objective=partial(train_minst_with_checkpoint, data_dir=data_dir),
        config=config,
        hyperparameters=config,
        metric='test_accuracy',
        mode='max',
        n_samples=20,
        max_iter=10,
        checkpoint_keep_num=7,
        perturbation_interval=2,
        cpus_per_trial=1,
        gpus_per_trial=0.2,
    )

    print("Best hyperparameters found were: ", results.get_best_result(metric="test_accuracy", mode="max"))
