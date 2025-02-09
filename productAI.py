import os
import numpy as np
import tensorflow as tf
from keras.api.applications import EfficientNetB0
from keras.api.preprocessing.image import load_img, img_to_array
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

def load_classes(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes

# Загрузка классов из файла
classes_file_path = 'food-101/meta/classes.txt'
food_classes = load_classes(classes_file_path)

# Подготовка модели
def create_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model

model = create_model(len(food_classes)+1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Загрузка предобученных весов
model.load_weights('food101.weights.h5')  # Укажите путь к обученной модели

# Функция для оценки веса еды
def estimate_weight(image_path):
    # Пример: оценка веса на основе площади изображения
    # Для реальной оценки можно использовать сегментацию (например, U-Net)
    image = Image.open(image_path)
    image_size = image.size[0] * image.size[1]  # Площадь в пикселях
    reference_area = 224 * 224  # Эталонная площадь (например, размер тарелки)
    weight = (image_size / reference_area) * 100  # Вес пропорционален эталону
    return weight

# FastAPI сервер
app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Сохранение изображения
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Предобработка изображения
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Распознавание еды
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    print(predictions)
    print(int(predicted_class[0]))
    print(food_classes[predicted_class[0]])
    return JSONResponse(content={
        "food": predictions.tolist()
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)