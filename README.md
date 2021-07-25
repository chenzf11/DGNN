# README

### 基于DGNN的人体姿态序列分类

### 环境要求（Dependencies）

Python >= 3.5

scipy >= 1.3.0

numpy >= 1.16.4

Pytorch >= 1.1.0

tensorboardX >= 1.8 (For logging)

极链AI云平台的Tesla T4 70GPU云服务器



### 权值文件下载以及数据准备（Downloading & Preparations）

下载数据集 “人体姿态序列分类.zip”

链接：[https://pan.baidu.com/s/18XpTD8Y97QAiQuc1tWJ1Mw ](https://pan.baidu.com/s/18XpTD8Y97QAiQuc1tWJ1Mw )

提取码：dsuj

解压后，将 “人体姿态序列分类 ” 文件夹放在 ‘./data/'目录下

下载已有的权值文件 “HumSk_dgnn-210-756.pt”

链接：[https://pan.baidu.com/s/1S2faxXzoZNaT4QAiHrUQFg ](https://pan.baidu.com/s/1S2faxXzoZNaT4QAiHrUQFg )

提取码：kl35

将权值文件放置在'./runs/'目录下



### 训练（Training）

请依次执行下列命令

cd data_gen

python Hum_gen_joint_data.py

python Hum_gen_bone_data.py

python main.py --config ./config/Humen-skele/train.yaml



### 测试（Testing）

python main.py --config ./config/Humen-skele/test.yaml



### 参数说明

Accuracy：分类准确度

Top1：每个batch正确分类的样本占总样本的比例（即分类准确度）



### 联系我们

2591733445@qq.com

