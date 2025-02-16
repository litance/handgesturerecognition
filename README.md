# handgesturerecognition
Finger number recognition system based on Tkinter data collection and MobileNetV2 基于 Tkinter 数据采集与 MobileNetV2 的手指数量识别系统

### 所需要第三方库 Third-party libraries needed to run this program
- numpy
- opencv
- tkinter
- pillow
- mediapipe
- tensorflow

## 1.首先, 安装第三方库 Install Third-party libraries
```
pip install numpy opencv-python tkinter pillow
pip install mediapipe
pip install tensorflow
```

## 2.下载模型 Download model
```
https://github.com/litance/handgesturerecognition.git
```

## 2.配置好代码 Configure the code(main.py)
### dataset.py
在这里，你可以制作你自己的dataset.
> Here you can create your own dataset.

当然，你也可以下载外部的dataset，只需要把文件格式放置成这样.
> Of course, you can also download external datasets, just put the file format like this.
```
dataset/
├── 0
├── 1
├── 2
├── 3
├── 4
└── 5
```

### train.py

batch_size(line:12)
```
batch_size = 512
```

epoch(line:46)
```
model.fit(train_ds, epochs=100)
```

## 3.运行 Run this program(main.py)
按q以结束进程
> Press q to end the process


