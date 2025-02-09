import os
import numpy as np
import json
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.applications import EfficientNetB0
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Путь к датасету
DATASET_DIR = 'food-101/'
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
META_DIR = os.path.join(DATASET_DIR, 'meta')


# Загрузка классов
def load_classes(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


food_classes = load_classes(os.path.join(META_DIR, 'classes.txt'))
num_classes = len(food_classes)


# Функция для подготовки данных из JSON
def prepare_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    filepaths = []
    labels = []
    for class_name, images in data.items():
        for img in images:
            filepaths.append(os.path.join(IMAGE_DIR, img + '.jpg'))
            labels.append(food_classes.index(class_name))
    return np.array(filepaths), np.array(labels)


train_filepaths, train_labels = prepare_data(os.path.join(META_DIR, 'train.json'))
test_filepaths, test_labels = prepare_data(os.path.join(META_DIR, 'test.json'))


# Функция генерации данных
def data_generator(filepaths, labels, batch_size=32, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    while True:
        for i in range(0, len(filepaths), batch_size):
            batch_paths = filepaths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = [tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
            ) for img in batch_paths]
            yield np.array(images), tf.keras.utils.to_categorical(batch_labels, num_classes)


# Генераторы
batch_size = 32
train_gen = data_generator(train_filepaths, train_labels, batch_size, augment=True)
test_gen = data_generator(test_filepaths, test_labels, batch_size, augment=False)


# Создание модели
def create_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    base_model.trainable = False  # Заморозка базовой модели

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)  # Dropout для борьбы с переобучением
    output = Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model


model = create_model(num_classes)

# Компиляция модели
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Коллбэки
checkpoint = ModelCheckpoint('food101_improved_test_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Параметры обучения
steps_per_epoch = len(train_filepaths) // batch_size
validation_steps = len(test_filepaths) // batch_size

# Обучение модели
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# Разморозка последних слоев и дообучение
for layer in model.layers[-30:]:
    layer.trainable = True

# Компиляция с более низким learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    epochs=30, #epochs=60,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)
