# 项目简介

这个项目中我使用了 ShuffleNetV2 模型完成了玉米叶片病斑图片的分类任务。

其中各个文件的作用：

| 文件名 | 作用 |
|---|---|
| model.py | 定义了 ShuffleNetV2 模型 |
| train.py | 模型训练文件 |
| main.py | 批量识别的可视化界面 |
| test.py | 单个图片的识别 |
| shufflenetv2.pth | 权重文件即模型 |
| test.jpg | 图片样例，供test.py使用 |

# ShuffleNetV2 模型概述

ShuffleNetV2 是一种轻量级卷积神经网络（CNN）架构，专为高效的移动和嵌入式设备上的图像分类任务而设计。与其前身 ShuffleNetV1 相比，ShuffleNetV2 提供了更高的计算效率和更好的性能。

## 关键特性

1. **通道分裂（Channel Split）**：
   - 在每个 ShuffleNet 单元中，将输入特征图分成两部分，一部分通过复杂的变换，另一部分保持不变。这种方法减少了计算量。

2. **通道洗牌（Channel Shuffle）**：
   - 通过洗牌操作，确保跨组的特征能够相互通信，提高了模型的表达能力。

3. **轻量级卷积操作**：
   - 采用深度可分离卷积（Depthwise Separable Convolution）和逐点卷积（Pointwise Convolution），显著减少了参数量和计算量。

4. **高效的下采样模块**：
   - 设计了一种新的高效下采样模块，能够在保持模型轻量化的同时，提高下采样的效果。

## 架构

ShuffleNetV2 的基本单元包括以下组件：

1. **通道分裂**：
   - 输入特征图被分成两部分，一部分通过复杂的变换，另一部分保持不变。

2. **复杂变换**：
   - 通过一系列卷积操作，包括 1x1 卷积、3x3 深度可分离卷积和 Batch Normalization。

3. **通道洗牌**：
   - 将变换后的特征图与保持不变的特征图进行通道洗牌，以确保跨组特征能够相互通信。

4. **拼接**：
   - 将洗牌后的特征图与保持不变的特征图拼接在一起，形成输出特征图。

## 应用场景

ShuffleNetV2 主要应用于需要高效计算的场景，如移动设备和嵌入式系统。其轻量级的特性使得它非常适合在资源受限的设备上进行实时图像分类和其他计算机视觉任务。

## 性能

与其他轻量级模型（如 MobileNetV2）相比，ShuffleNetV2 在相同的计算预算下提供了更高的准确性。其创新的架构设计和高效的计算操作使其成为移动和嵌入式设备上的理想选择。

## 参考资料

- [ShuffleNetV2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
- [GitHub Repository](https://github.com/megvii-model/ShuffleNet-Series)


