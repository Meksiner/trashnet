import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

# Папка для сохранения загруженных изображений
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Загрузка модели
MODEL_PATH = "trashnet_classifier.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Ошибка: файл модели '{MODEL_PATH}' не найден!")

model = tf.keras.models.load_model(MODEL_PATH)

# Классы
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Загружает и предобрабатывает изображение."""
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_trash_class(img_path):
    """Выполняет предсказание класса мусора."""
    processed_img = load_and_preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "Ошибка: файл не был загружен!", 400

        file = request.files["file"]
        if file.filename == "":
            return "Ошибка: имя файла пустое!", 400

        # Путь сохранения файла
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Предсказание класса
        predicted_class, confidence = predict_trash_class(file_path)

        return render_template("result.html",
                               predicted_class=predicted_class,
                               confidence=confidence)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
