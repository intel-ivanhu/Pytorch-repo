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
    

    # load image
    img = Image.open("1.jpg")

    width, height = img.size
    print(f"Width: {width}, Height: {height}")
    #img = img.Resize(224,224)
    # [N, C, H, W]
    #tensorizer = ToTensor()
    img_tensor = transform(img)
    
    print(img_tensor.shape) #(3,32,32)
    # expand batch dimension
    #img_tensor = torch.unsqueeze(img_tensor, dim=0)		# 增加一个 batch 维度
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)

    with torch.no_grad():
        outputs = net(img_tensor)
        #outputs.type()
        #predict = torch.max(outputs, dim=1)[1].numpy()
    #print(classes[int(predict)])   

    #out_put = net(img)
    #outputs
    #print(outputs.shape)
    #print(outputs)
    #print(outputs[1].shape)
    for feature_map in outputs:			
        # [N, C, H, W] -> [C, H, W]
        print("feature_map")
        print(feature_map.shape) 

        im = np.squeeze(feature_map.detach().numpy())		# 只输入了一张图，squeeze 压缩掉 batch 维度，detach() 去除梯度信息
        
        print(im.shape) # 16 28*28
        #pritn(im.)
        # [C, H, W] -> [H, W, C]
        im_hwc = np.transpose(im, [1, 2, 0])
        print(im_hwc.shape)
        
        # show top 16 feature maps
        plt.figure()
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            # [H, W, C]
            # 我们特征矩阵每一个 channel 所对应的是一个二维的特征矩阵，就像灰度图一样，channel = 1
            # 如果不指定 cmap='gray' 默认是以蓝色和绿色替换黑色和白色
            plt.imshow(im_hwc[:, :, i])
        plt.show()



if __name__ == '__main__':
    main()
