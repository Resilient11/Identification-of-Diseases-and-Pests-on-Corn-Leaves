import torch
from model import shufflenetv2
from torchvision import transforms
from PIL import Image
from train import trainset


# print(trainset.classes)    


# 加载模型
model = shufflenetv2(num_classes=4)
model.load_state_dict(torch.load('shufflenetv2.pth'))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    output = model(image)
    predicted_label_idx = torch.argmax(output, dim=1).item()
    predicted_label_name = trainset.classes[predicted_label_idx]
    return predicted_label_name


if __name__ == '__main__':
    image_path = 'test.jpg'
    predicted_label = predict_image(image_path)
    print(f"Predicted label: {predicted_label}")
