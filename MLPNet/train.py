import torch
import torchvision
import torchvision.transforms as tranforms
from matplotlib import pyplot as plt
import numpy as np
from model import SimpleNet

def main():
    data_dir = './fashion_mnist/'
    tranform = tranforms.Compose([tranforms.ToTensor()])

    train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform, download=False)
    print("train_dataset nums:",len(train_dataset))

    val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform, download=False)
    print("val_dataset nums:",len(val_dataset))

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("train_loader nums:",len(train_loader))
    print("test_loader nums:",len(test_loader))


    CNN_Net = SimpleNet()
    print(CNN_Net)

    device_count = torch.cuda.device_count()
    print(device_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    CNN_Net.to(device)

    #loss function
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(CNN_Net.parameters(), lr=0.001)

    for epoch in range(1): #数据集迭代2次
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0): #循环取出批次数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) #
            optimizer.zero_grad()#清空之前的梯度
            outputs = CNN_Net(inputs)
            loss = criterion(outputs, labels)#计算损失
            loss.backward()  #反向传播
            optimizer.step() #更新参数
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
        print('Finished Training')
        # 保存模型
        torch.save(CNN_Net.state_dict(), './CNNFashionMNIST.pth')

if __name__ == '__main__':
    main()
