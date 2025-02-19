import sys
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from torchvision import transforms
from model import shufflenetv2


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化模型
        self.model = shufflenetv2(num_classes=4)  # 这里使用的模型结构
        self.model.load_state_dict(torch.load('shufflenetv2.pth'))  # 加载保存的权重
        self.model.eval()

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 800, 600)  # 设置窗口大小
        self.setMinimumSize(400, 300)  # 设置最小窗口尺寸

        # 创建控件
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel('请选择一张图片进行分类', self)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.upload_button = QPushButton('上传图片', self)
        self.upload_button.clicked.connect(self.upload_image)

        self.classify_button = QPushButton('进行分类', self)
        self.classify_button.clicked.connect(self.classify_image)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def upload_image(self):
        # 打开文件对话框选择图片
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)")

        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)  # 适应显示框大小
            self.image_label.setPixmap(pixmap)
            self.result_label.setText('点击 “进行分类” 按钮来进行预测')

    def classify_image(self):
        if hasattr(self, 'image_path'):
            # 读取图片并进行预处理
            image = Image.open(self.image_path)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)  # 增加批次维度

            # 使用模型进行推理
            with torch.no_grad():
                output = self.model(image_tensor)

            # 获取预测结果
            _, predicted = torch.max(output, 1)

            # 获取类别名称
            class_names = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight']  # 你自己的类别名称
            predicted_class_name = class_names[predicted.item()]

            # 显示结果
            self.result_label.setText(f"预测类别：{predicted_class_name}")
        else:
            self.result_label.setText("请先上传图片！")

    def resizeEvent(self, event):
        # 控件随窗口尺寸调整
        new_size = self.size()
        self.upload_button.setFixedWidth(new_size.width() // 4)
        self.classify_button.setFixedWidth(new_size.width() // 4)
        self.image_label.setFixedHeight(new_size.height() // 2)
        super().resizeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())


# 显示结果
# class_names = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight']  # 你的类别名称
