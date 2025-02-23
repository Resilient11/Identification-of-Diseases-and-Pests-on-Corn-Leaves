第三版
=====

（其实也是最终版）

在第二版中，最终的分析都错了......真正的原因是在训练集和测试集以及使用模型时的transforms.Resize(())中的参数不一致导致的。

另外我还发现了我之前并没有使用GPU来训练。那么在这一版中，我配置了cuda环境，使用了GPU加速训练，并调高了transforms.Resize(())的参数值，重新训练了一遍。

再一次得到了如下结果：

| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | **97.09% Gray_leaf_spot**、00.00% Common_rust、2.91% Healthy、0% Leaf_Blight|
| Common_rust     | 0.42% Gray_leaf_spot、**99.58% Common_rust**、0% Healthy、0% Leaf_Blight |
| Healthy         | 0.43% Gray_leaf_spot、0.00% Common_rust、0.00% Healthy、**99.57% Leaf_Blight** |
| Leaf_Blight     | 5.01% Gray_leaf_spot、0.51% Common_rust、**94.39% Healthy**、0.00% Leaf_Blight|

这样的结果还不准确，但是比之前版本的好太多了。

虽然后两类分类不正确，但是所有类别都基本都倾向于同一个类。再加上模型的 Best Accuracy为97.9221% 这表明模型是没有问题的。

那么说明是在使用模型时，标签对应有问题。

运行以下代码，获取模型训练时的标签列表：

  ``print(trainset.classes)``

得到：['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy']

这与数据集中文件夹的排序并不相同：

Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot

Corn_(maize)___Common_rust_

Corn_(maize)___healthy

Corn_(maize)___Northern_Leaf_Blight

个人**猜测**是因为在使用torchvision.datasets.ImageFolder加载数据集时（也就是上面print(trainset.classes)的结果），按照"大写字母>小写字母"的排序原则，而 Windows 文件资源管理器按照字母顺序优先的原则，才导致这样的结果。

于是修改的main.py中的class_names = ['Gray_leaf_spot', 'Common_rust', 'Leaf_Blight', 'Healthy']就大功告成了！

附上最终模型的识别结果：

| 真实类别 | 识别结果 |
|-----|-----|
| Gray_leaf_spot  | **97.09% Gray_leaf_spot**、0% Common_rust、0% Healthy、2.91% Leaf_Blight|
| Common_rust     | 0.42% Gray_leaf_spot、**99.58% Common_rust**、0% Healthy、0% Leaf_Blight |
| Healthy         | 0.43% Gray_leaf_spot、0.00% Common_rust、**99.57% Healthy**、0% Leaf_Blight |
| Leaf_Blight     | 5.01% Gray_leaf_spot、0.51% Common_rust、0% Healthy、**94.39% Leaf_Blight**|

