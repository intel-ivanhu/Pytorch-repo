import torch
import torchvision
import torch.nn as nn
from model_LeNet import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    net = LeNet()
    print(net)
    #device_count = torch.cuda.device_count()
    #print(device_count)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #net_weight_path = "./Lenet.pth"  # "./resNet34.pth"
    net.load_state_dict(torch.load('Lenet.pth',weights_only=True))
    
    weights_keys = net.state_dict().keys()

    print(weights_keys)

    for key in weights_keys:
        # remove num_batches_tracked para(in bn)
        if "num_batches_tracked" in key:
            continue
        print(key)
        weight_t = net.state_dict()[key].numpy()
        #print(weight_t)
        weight_mean = weight_t.mean()
        weight_std = weight_t.std(ddof=1)
        weight_min = weight_t.min()
        weight_max = weight_t.max()
        print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))
        plt.close()
        weight_vec = np.reshape(weight_t, [-1])		# 卷积核权重展成一维的向量 --- 原始卷积核太小了就3x3
        plt.hist(weight_vec, bins=50)							# 统计卷积核权重值直方图的分布
        plt.title(key)
        plt.show()


if __name__ == '__main__':
    main()
