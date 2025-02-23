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
from torch.amp import autocast, GradScaler
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="shufflenetv2", type=str)
parser.add_argument('--dataset', default="test-data", type=str, help="cifar100|cifar10")
parser.add_argument('--epoch', default=50, type=int, help="training epochs")
parser.add_argument('--dataset_path', default="input", type=str)
parser.add_argument('--batchsize', default=64, type=int)  # 减小批量大小
parser.add_argument('--init_lr', default=0.01, type=float)
args = parser.parse_args(args=[])



def GetMetrics(all_predicted_labels, all_true_labels):
    sw = compute_sample_weight(class_weight='balanced', y=all_true_labels)

    precision = precision_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw, zero_division=0)
    recall = recall_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw, zero_division=0)
    f1 = f1_score(all_predicted_labels, all_true_labels, average='weighted', sample_weight=sw, zero_division=0)

    return [precision, recall, f1]


class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


if args.dataset == "test-data":
    train_dir = './input/test-data/train'
    test_dir = './input/test-data/test'
    num_classes = 4

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = datasets.ImageFolder(train_dir, transform=transform_train)

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testset = datasets.ImageFolder(test_dir, transform=transform_test)

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
scaler = GradScaler()

if __name__ == "__main__":
    best_acc = 0
    all_true_labels = []

    print(f"使用的设备: {device}")
    print(args)

    for images, targets in testloader:
        all_true_labels.extend(targets.cpu().numpy().tolist())
    all_true_labels = np.array(all_true_labels)

    best_metrics = [0.0, 0.0, 0.0]
    for epoch in range(args.epoch):
        if (epoch + 1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type='cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 清理未使用的变量
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        all_predicted_labels = []
        with torch.no_grad():
            correct = 0
            total = 0.0

            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

                all_predicted_labels.extend(predicted.cpu().numpy().tolist())

        all_predicted_labels = np.array(all_predicted_labels)
        acc = 100 * correct / total
        print(f'Epoch {epoch + 1} Test Set Accuracy: {acc:.4f}%')

        if correct / total > best_acc:
            best_acc = correct / total
            print(f'Best Accuracy Updated: {best_acc * 100:.4f}%')
            torch.save(net.state_dict(), f"{args.model}.pth")

        result = GetMetrics(all_predicted_labels, all_true_labels)
        print(f'精确率为: {result[0]:.4f}, 召回率为: {result[1]:.4f}, F1值为: {result[2]:.4f}')
        for i in range(3):
            if result[i] > best_metrics[i]:
                best_metrics[i] = result[i]

    print(f"Training Finished, Total Epochs={args.epoch}, Best Accuracy={best_acc * 100:.4f}")
    print(f'精确率为: {best_metrics[0]:.4f}, 召回率为: {best_metrics[1]:.4f}, F1值为: {best_metrics[2]:.4f}')