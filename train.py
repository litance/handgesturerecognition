import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import os

# 数据集路径
dataset_dir = "dataset"

# 预处理函数
img_size = (128, 128)
batch_size = 256

train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int"
)

# 加载 MobileNetV2 作为特征提取器
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # 冻结预训练权重

# 构建分类模型
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(6, activation="softmax")  # 6 类（0-5 手指）
])

# 编译模型
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 训练模型
model.fit(train_ds, epochs=10)

# 保存模型
model.save("finger_count_mobilenetv2.h5")
