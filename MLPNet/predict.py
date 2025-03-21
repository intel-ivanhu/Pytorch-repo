import torch
import torchvision
import torchvision.transforms as tranforms
from matplotlib import pyplot as plt
import numpy as np
from model import SimpleNet

def imshow(img):
    print("shape img:",np.shape(img))
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():
    network = SimpleNet()

    data_dir = './fashion_mnist/'
    tranform = tranforms.Compose([tranforms.ToTensor()])

    #train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform, download=False)
    #print("train_dataset nums:",len(train_dataset))

    val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform, download=False)
    print("val_dataset nums:",len(val_dataset))

    batch_size = 10
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    network.load_state_dict(torch.load( './CNNFashionMNIST.pth'))
    
    #
    classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs, labels = images.to(device), labels.to(device)
    
    imshow(torchvision.utils.make_grid(images,nrow=batch_size))
    
    

    print('y class: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    outputs = network(inputs)
    _, predicted = torch.max(outputs, 1)
    
    
    print('results: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(len(images))))
    
    
    #测试模型
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            inputs, labels = images.to(device), labels.to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    sumacc = 0
    for i in range(10):
        Accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy ))
        sumacc =sumacc+Accuracy
    print('Accuracy of all : %2d %%' % ( sumacc/10. ))

if __name__ == '__main__':
    main()
