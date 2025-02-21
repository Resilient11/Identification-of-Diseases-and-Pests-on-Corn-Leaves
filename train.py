import torch
import torch.nn.functional as F
import PIL.Image
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import compute_sample_weight
from ptflops import get_model_complexity_info
import argparse

from model import shufflenetv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="shufflenetv2", type=str)
parser.add_argument('--dataset', default="test-data", type=str, help="cifar100|cifar10")
parser.add_argument('--epoch', default=100, type=int, help="training epochs")
parser.add_argument('--dataset_path', default="input", type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.01, type=float)
args = parser.parse_args(args=[])
print(args)


def GetMetrics(all_predicted_labels, all_true_labels):
    sw = compute_sample_weight(class_weight='balanced', y=all_true_labels)

    # 设置 zero_division 参数来处理除零错误
    precision = precision_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw,
                                zero_division=0)
    recall = recall_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw, zero_division=0)
    f1 = f1_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw, zero_division=0)

    return [precision, recall, f1]


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= PIL.Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img
    
if args.dataset == "test-data":
    train_dir = './input/test-data/train'
    test_dir = './input/test-data/test'
    num_classes = 4

transform_train = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation(60),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5)
    ], p=0.5),
    AddSaltPepperNoise(0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = datasets.ImageFolder(train_dir,transform=transform_train)

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testset = datasets.ImageFolder(test_dir,transform=transform_test)
    
    
trainloader = DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=2
)
testloader = DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=2
)

if args.model == "shufflenetv2":
    net = shufflenetv2(num_classes=num_classes)

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)

# num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True, verbose=True)
# print(macs)
# print(num_params)

if __name__ == "__main__":
    best_acc = 0
    all_true_labels = []  # 存储所有真实标签
    for images,targets in testloader:
        all_true_labels.extend(targets.cpu().numpy().tolist())
    all_true_labels = np.array(all_true_labels)
    
    best_metrics = [0.0, 0.0, 0.0]
    for epoch in range(args.epoch):
        if (epoch+1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = torch.FloatTensor([0.]).to(device)
            loss += criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        all_predicted_labels = []  # 存储所有的预测结果
        with torch.no_grad():
            correct = [0]
            predicted = [0]
            total = 0.0
            
            for data in testloader:
                net.eval()
                images, labels = data
                true_labels = labels
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted[0] = torch.max(outputs.data, 1)
                correct[0] += float(predicted[0].eq(labels.data).cpu().sum())
                total += float(labels.size(0))
                
                all_predicted_labels.extend(predicted[0].cpu().numpy().tolist())

        all_predicted_labels = np.array(all_predicted_labels)
        print('Epoch{} Test Set AccuracyAcc: {:.4f}'.format(epoch+1, 100*correct[0]/total))
        
        if correct[0] / total > best_acc:
            best_acc = correct[0]/total
            print('Best Accuracy Updated: {:.4f}'.format(best_acc * 100))
            torch.save(net.state_dict(), str(args.model)+".pth")
        result = GetMetrics(all_predicted_labels,all_true_labels)
        print('精确率为{:.4f}, 召回率为{:.4f}, F1值为{:.4f}'.format(result[0],result[1],result[2]))
        for i in range(3):
            if result[i] > best_metrics[i]:
                best_metrics[i] = result[i]
    print("Training Finished, TotalEpoch=%d, Best Accuracy=%.4f" % (args.epoch, best_acc))
    print('精确率为{:.4f}, 召回率为{:.4f}, F1值为{:.4f}'.format(best_metrics[0],best_metrics[1],best_metrics[2]))
