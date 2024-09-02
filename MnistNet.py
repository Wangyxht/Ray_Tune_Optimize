from torch import nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    """
    MINST全连接网络结构
    :param l1: 第一个隐藏层的节点数
    :param l2: 第二个隐藏层的节点数
    """
    def __init__(self, l1: int, l2: int):
        super(MnistNet, self).__init__()
        self.layer1 = nn.Linear(in_features=28 * 28, out_features=l1)
        self.layer2 = nn.Linear(in_features=l1, out_features=l2)
        self.layer3 = nn.Linear(in_features=l2, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
