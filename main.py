import sys
import torch
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from torchvision import transforms
from PIL import Image
from model import shufflenetv2

class ClassifyThread(QThread):
    # Define custom signals to update UI
    update_result_signal = pyqtSignal(str)
    update_summary_signal = pyqtSignal(str)

    def __init__(self, folder_path, model, transform):
        super().__init__()
        self.folder_path = folder_path
        self.model = model
        self.transform = transform
        self.class_count = {'Gray_leaf_spot': 0, 'Common_rust': 0, 'Healthy': 0, 'Leaf_Blight': 0}
        self.total_images = 0

    def run(self):
        result = ""
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    output = self.model(img_tensor)
                    _, predicted = torch.max(output, 1)

                # Get predicted class label
                class_names = ['Gray_leaf_spot', 'Common_rust', 'Healthy', 'Leaf_Blight']  # Modify according to your class names
                predicted_class = class_names[predicted.item()]

                # Update the class count
                self.class_count[predicted_class] += 1
                self.total_images += 1

                result += f"Image: {filename}, Predicted Class: {predicted_class}\n"

        # Calculate percentages and send to UI
        summary = "\nClassification Summary:\n"
        for class_name, count in self.class_count.items():
            percentage = (count / self.total_images) * 100 if self.total_images > 0 else 0
            summary += f"{class_name}: {count} images ({percentage:.2f}%)\n"

        self.update_result_signal.emit(result)  # Update UI with result
        self.update_summary_signal.emit(summary)  # Update UI with summary


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Batch Image Classifier")
        self.setGeometry(100, 100, 600, 400)

        self.layout = QVBoxLayout()

        # Add widgets
        self.label = QLabel("Choose a folder containing images", self)
        self.layout.addWidget(self.label)

        self.text_edit = QTextEdit(self)
        self.layout.addWidget(self.text_edit)

        self.choose_folder_button = QPushButton("Choose Folder", self)
        self.choose_folder_button.clicked.connect(self.choose_folder)
        self.layout.addWidget(self.choose_folder_button)

        self.classify_button = QPushButton("Classify Images", self)
        self.classify_button.clicked.connect(self.classify_images)
        self.layout.addWidget(self.classify_button)

        self.setLayout(self.layout)

        # Load the model
        self.model = shufflenetv2(num_classes=4)
        self.model.load_state_dict(torch.load('shufflenetv2.pth'))
        self.model.eval()

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.text_edit.append(f"Selected folder: {folder}")
            self.folder_path = folder
        else:
            self.text_edit.append("No folder selected.")

    def classify_images(self):
        if hasattr(self, 'folder_path'):
            self.text_edit.append("Classifying images...")

            # Start the classification in a new thread
            self.classify_thread = ClassifyThread(self.folder_path, self.model, self.transform)

            # Connect signals to update the UI
            self.classify_thread.update_result_signal.connect(self.update_result)
            self.classify_thread.update_summary_signal.connect(self.update_summary)

            # Start the thread
            self.classify_thread.start()
        else:
            self.text_edit.append("Please select a folder first.")

    def update_result(self, result):
        self.text_edit.append(result)

    def update_summary(self, summary):
        self.text_edit.append(summary)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec_())
