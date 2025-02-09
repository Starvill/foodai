import os
import numpy as np
import pandas as pd  # Импорт pandas для работы с DataFrame
import tensorflow as tf
from keras.api.callbacks import ModelCheckpoint
from keras.api.applications import EfficientNetB0
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D, Input
from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import TensorBoard

# Укажите путь к датасету
data_dir = "food-101/images"

# Параметры
IMG_SIZE = (224, 224)  # Размер изображения для EfficientNet
BATCH_SIZE = 32  # Размер батча


# Функция для подготовки данных
def prepare_data(data_dir, test_size=0.2):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    images = []
    labels = []

    # Пропуск файлов, не являющихся папками или изображениями
    for idx, label in enumerate(classes):
        class_dir = os.path.join(data_dir, label)

        # Пропускаем скрытые файлы и папки
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            # Пропускаем файлы, которые не являются изображениями
            if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, img_name)
            images.append(img_path)
            labels.append(idx)

    # Разделение на обучающую и тестовую выборки
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )

    return train_images, val_images, train_labels, val_labels, classes


train_images, val_images, train_labels, val_labels, food_classes = prepare_data(data_dir)

# Создание модели EfficientNet
def create_model(num_classes):
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = create_model(len(food_classes))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Преобразуем данные в pandas DataFrame
train_df = pd.DataFrame({"images": train_images, "labels": train_labels})
val_df = pd.DataFrame({"images": val_images, "labels": val_labels})

# Создание генераторов данных
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="images",
    y_col="labels",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="images",
    y_col="labels",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

checkpoint = ModelCheckpoint(
    filepath="food101_weights_epoch_{epoch:02d}.weights.h5",
    save_weights_only=True,  # сохранять только веса модели
    save_freq='epoch',       # сохранять после каждой эпохи
    verbose=1                # выводить информацию о сохранении
)

# Обучение модели
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint]
)

# Сохранение весов
model.save_weights("food101.weights.h5")
