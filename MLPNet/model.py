import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        #定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        #self.bn1 = torch.nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        #定义全连接层
        self.fc1 = nn.Linear(in_features=12*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):#搭建正向结构
        #第一层卷积和池化处理
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = self.bn1(t)

        #第二层卷积和池化处理
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        #搭建全连接网络，第一层全连接
        t = t.reshape(-1, 12 * 5 * 5)#将卷积结果由4维变为2维
        t = self.fc1(t)
        t = F.relu(t)

        #第二层全连接
        t = self.fc2(t)
        t = F.relu(t)

        #第三层全连接
        t = self.out(t)
        return t
