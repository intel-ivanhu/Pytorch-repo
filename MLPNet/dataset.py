import torchvision
import torchvision.transforms as tranforms

data_dir = './fashion_mnist/'
tranform = tranforms.Compose([tranforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform,download=True)
print("训练数据集条数",len(train_dataset))

val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform,download=True)
print("测试数据集条数",len(val_dataset))