# captcha_trainer_pytorch

## 项目介绍

不定长验证码识别训练工具，基于Pytorch 1.6开发，神经网络结构为: CNN + LSTM + CTC。

本项目主要用于不定长验证码的训练，包含有模型预测推理的demo.

支持CPU/GPU训练，CPU训练速度缓慢,GPU训练速度约为CPU的50倍.

训练完成后部署可使用CPU，可无需使用GPU,CPU识别速度约为10-25ms.

项目参考: https://github.com/ghosthamlet/captcha_trainer .

使用提醒：使用本项目默认认为您已经具备了必要pytorch安装知识，python基础开发能力或有一定辨别错误类型或调试的能力。

## 项目使用手册

### 1. 初始化项目目录

```shell script
python main.py init <project_name>

example:
    python main.py init test_framework
```

本条命令中<project_name>为您的项目名称，如您的项目名称为test_framework，则运行上面example中的命令

其中<project_name>为必填参数.

### 2. 导入数据集

```shell script
python main.py data <project_name> <images_path> <scale=0.97> <words=False>

example:
    python main.py data test_framework D:\images
    python main.py data test_framework /mnt/images
```

本命令背后的支持模块仅支持 jpg、png、jpeg和bmp格式的图片数据，且统一存放在<images_path>目录下，样本需要按照abcd_xx.jpg的格式存放，abcd为您图像的具体标签，xx为任意随机值，用于区分同一标签的不同图像。

运行本命令后，工具会自动按照参数scale的值进行比例切割样本集为训练集和测试集，并在过程中检测图像的合法性，结果将被导出为两个包含有图像路径的json文件中。

scale参数应当小于1且大于0，数值代表的是训练集数目占全体样本数目的比例.

words参数为是否保留标签为一个单词整体，如abcd_xx.jpg命名的样本，默认words为False时标签将会被看作["a", "b", "c", "d"]，如words为True时，标签将被视为["abcd"]

参数中project_name和images_path为必填，scale和words非必填

### 3. 训练项目

```shell script
python main.py train <project_name>

example:
    python main.py train test_framework
```
这里就没啥好说的了~~~

训练过程中会在projects中对应项目名称的文件夹中的models文件夹生成pkl文件，用于训练中断后的恢复以及保存训练状态。

训练完成后会在graphs目录下生成onnx模型，在下一步中提供了调用onnx模型进行预测推理的demo，也可以自行研究移植到其他平台下调用。

### 4. 推理预测

修改server.py

```shell script
python server.py
```

### 5. 配置项讲解

经过第1步，在projects目录下会生成对应项目名称的文件夹，其中会生成一个config.yaml文件。

```yaml
Model:
  CharSet: '["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d",
    "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"]' # 默认字符集，无需修改，在导入样本后会自动更新此项
  ImageChannel: 1 # 训练时期待的图像通道数，1为黑白，3为彩图
  ImageHeight: 50 # 原始图片的高度，其实改不改这里无所谓，暂时用不上
  ImageWidth: 150 # 原始图片的宽度，其实改不改这里无所谓，暂时用不上
  RESIZE: # 图像尺寸归一化
  - 150 # RESIZE 宽度
  - 50  # RESIZE 高度
  Word: false # 标签切割方式，配置里的这个暂时用不上
System:
  GPU: true # 是否启用GPU进行训练
  GPU_ID: 0 # 机器有多卡时指定用于训练的GPU的ID，默认从0开始
  Project: test_framework # 项目名称
Train:
  BATCH_SIZE: 32 # 训练时一个BATCH中有多少张图
  CNN:
    NAME: MobileNetV2 # 特征提取层的神经网络名称，目前支持有MobileNetV2和EfficientNet-b0
  LR: 0.01 # 学习率
  LSTM:
    DROPOUT: 0.8
    HIDDEN_NUM: 64
  OPTIMIZER: Momentum # 优化器，目前支持有Momentum以及Adam
  RNN:
    NAME: LSTM
  TARGET:
    Accuracy: 0.97 # 训练目标准确率,1为100%正确
    Cost: 0.005 # 目标损失率
    Epoch: 200
  TEST_BATCH_SIZE: 32 # 测试时一个BATCH中有多少张图
  TEST_STEP: 1000 

```

当前版本目前支持的特征提取层的网络为 MobileNetV2、EfficientNet-b0

![qcc_tensorflow_trainer](https://ss0.bdstatic.com/-0U0bnSm1A5BphGlnYG/tam-ogel/1d3df1d8e4b4ca973ca5e8d0cbe84c26_121_121.jpg)

查企业就上企查查

文安哲

当前版本号: v0.01

交流QQ群：778910502
