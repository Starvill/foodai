import os
import numpy as np
import pandas as pd  # Импорт pandas для работы с DataFrame
import tensorflow as tf
from keras.api import models
from keras.api.callbacks import ModelCheckpoint
from keras.api.applications import EfficientNetB0
from keras.api import layers
from keras.api.layers import Dense, GlobalAveragePooling2D, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import TensorBoard
import json

from tensorflow.python.keras.callbacks import Callback

data_dir = "food-101/images"
meta_dir = "food-101/meta"

# Подготовка данных из train.json и test.json
def prepare_data(meta_dir, data_dir):
    # Загрузка метаданных
    with open(os.path.join(meta_dir, "train.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(meta_dir, "test.json")) as f:
        test_data = json.load(f)

    # Подготовка тренировочных данных
    train_paths, train_labels = [], []
    for label, images in train_data.items():
        for image in images:
            train_paths.append(os.path.join(data_dir, f"{image}.jpg"))
            train_labels.append(label)

    # Подготовка тестовых данных
    test_paths, test_labels = [], []
    for label, images in test_data.items():
        for image in images:
            test_paths.append(os.path.join(data_dir, f"{image}.jpg"))
            test_labels.append(label)

    return train_paths, train_labels, test_paths, test_labels

train_paths, train_labels, test_paths, test_labels = prepare_data(meta_dir, data_dir)

# Разделение тренировочных данных на train/val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Параметры для генератора данных
img_size = (224, 224)
batch_size = 32

# Генераторы данных
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
).flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": train_paths, "class": train_labels}),
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": val_paths, "class": val_labels}),
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse"
)

test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": test_paths, "class": test_labels}),
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse"
)

# Создание модели
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False  # Замораживаем базовую модель

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(set(train_labels)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

class UnfreezeBaseModel(Callback):
    def __init__(self, base_model, unfreeze_epoch):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch:
            print(f"Эпоха {epoch + 1}: Размораживаем базовую модель.")
            self.base_model.trainable = True
            # Перекомпилируем модель с новым состоянием trainable
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Добавляем коллбэк в список
unfreeze_callback = UnfreezeBaseModel(base_model=base_model, unfreeze_epoch=20)

# Callbacks
tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1)

checkpoint = ModelCheckpoint(
    filepath="food101_test_weights_epoch_{epoch:02d}.weights.h5",
    monitor="val_accuracy",
    save_weights_only=True,
    verbose=1,
    mode="max",
    save_best_only=True
)

# Обучение модели
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=[tensorboard, checkpoint, unfreeze_callback]
)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Сохранение модели
model.save_weights("food101_test.weights.h5")