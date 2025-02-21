# Identification-of-Diseases-and-Pests-on-Corn-Leaves
基于深度学习的玉米叶片病虫害识别系统(A corn leaf disease and pest identification system based on deep learning)

第二版
-----
在第二版中，使用批量识别的方式，对测试文件夹内所有的图片进行批量识别，同时将transforms.Resize((64,64)),修改为transforms.Resize((128,128))再次训练

识别结果如下：


在测试集上的识别情况
--------
 第一次：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.97% Gray_leaf_spot、00.00% Common_rust、**97.09% Healthy**、1.94% Leaf_Blight|
| Common_rust     | 0.29% Gray_leaf_spot、**69.50% Common_rust**、29.62% Healthy、0.59% Leaf_Blight |
| Healthy         | 0.17% Gray_leaf_spot、**41.29% Common_rust**、17.60% Healthy、**40.94% Leaf_Blight** |
| Leaf_Blight     | 0.13% Gray_leaf_spot、**30.91% Common_rust、37.53% Healthy、31.43% Leaf_Blight**|

第二次：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.97% Gray_leaf_spot、00.00% Common_rust、**97.09% Healthy**、1.94% Leaf_Blight|
| Common_rust     | 0.00% Gray_leaf_spot、**99.58% Common_rust**、0.42% Healthy、0.00% Leaf_Blight |
| Healthy         | 0% Gray_leaf_spot、0% Common_rust、 0% Healthy、**100% Leaf_Blight** |
| Leaf_Blight     | 0% Gray_leaf_spot、0.51% Common_rust、**95.92% Healthy**、3.57% Leaf_Blight |

第三次（与第二次完全相同）：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.97% Gray_leaf_spot、00.00% Common_rust、97.09% Healthy、1.94% Leaf_Blight|
| Common_rust     | 0.00% Gray_leaf_spot、99.58% Common_rust、0.42% Healthy、0.00% Leaf_Blight |
| Healthy         | 0% Gray_leaf_spot、0% Common_rust、 0% Healthy、100% Leaf_Blight |
| Leaf_Blight     | 0% Gray_leaf_spot、0.51% Common_rust、95.92% Healthy、3.57% Leaf_Blight |


在训练集上的识别情况
------
 第一次：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.24% Gray_leaf_spot、00.24% Common_rust、**97.07% Healthy**、2.44% Leaf_Blight|
| Common_rust     | 0.00% Gray_leaf_spot、**99.47% Common_rust**、0.21% Healthy、0.32% Leaf_Blight |
| Healthy         | 0.00% Gray_leaf_spot、0.00% Common_rust、0.00% Healthy、**100% Leaf_Blight** |
| Leaf_Blight     | 0.00% Gray_leaf_spot、0.13% Common_rust、**96.68% Healthy**、3.20% Leaf_Blight|

第二次（与第一次完全相同）：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.24% Gray_leaf_spot、00.24% Common_rust、**97.07% Healthy**、2.44% Leaf_Blight|
| Common_rust     | 0.00% Gray_leaf_spot、**99.47% Common_rust**、0.21% Healthy、0.32% Leaf_Blight |
| Healthy         | 0.00% Gray_leaf_spot、0.00% Common_rust、0.00% Healthy、**100% Leaf_Blight** |
| Leaf_Blight     | 0.00% Gray_leaf_spot、0.13% Common_rust、**96.68% Healthy**、3.20% Leaf_Blight|

第三次（与第一次完全相同）：
| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | 0.24% Gray_leaf_spot、00.24% Common_rust、**97.07% Healthy**、2.44% Leaf_Blight|
| Common_rust     | 0.00% Gray_leaf_spot、**99.47% Common_rust**、0.21% Healthy、0.32% Leaf_Blight |
| Healthy         | 0.00% Gray_leaf_spot、0.00% Common_rust、0.00% Healthy、**100% Leaf_Blight** |
| Leaf_Blight     | 0.00% Gray_leaf_spot、0.13% Common_rust、**96.68% Healthy**、3.20% Leaf_Blight|


结果分析：
------
遇到的主要问题是**类别不平衡** 和**模型过拟合**，导致模型倾向于预测为 Healthy 类，并且对其他类别的识别效果较差。解决该问题需要通过调整训练数据的平衡、增加数据增强、优化模型架构以及改进训练策略来提高模型的准确性和泛化能力。

PS:
---
在这次的代码中加入了**多线程**避免了GUI界面阻塞，并且使用了**批量识别**的方式以及程序**自动统计**分类数量比例来快速测试模型的准确率。
