评价一款葡萄酒时不外乎从颜色、酸度、甜度、香气、风味等入手，而决定这些就是葡萄酒的挥发酸度、糖分、密度等。

根据给出的白葡萄酒酸度、糖分、PH值、柠檬酸等数据，判断葡萄酒品质。

数据集下载

下载链接：https://static.leiphone.com/winequality_dataset.zip



数据集说明：

1、训练集字段

fixed acidity ; volatile acidity ; citric acid ; residual sugar ; chlorides ; free sulfur dioxide ; total sulfur dioxide ; density ; pH ; sulphates ; alcohol ; quality

2、以后比赛数据集会统一使用逗号“，”分隔（感谢追梦人的建议）

3、表头出现在训练集第630行，第630行结果全部不计入成绩



结果文件如下所示：

第一个字段位：测试集样本ID

第二个字段：葡萄酒品质得分

葡萄酒结果示例.png



评审标准

MAE：平均绝对误差，可以更好地反应预测与实际结果的误差情况。

MAE.png
https://god.yanxishe.com/15
