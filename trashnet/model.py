import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Путь к датасету
dataset = 'dataset-re'

# Загрузка данных
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset, validation_split=0.2, subset="training",
    seed=123, image_size=(224, 224), batch_size=64  # Уменьшаем батч для стабильности
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset, validation_split=0.2, subset="validation",
    seed=123, image_size=(224, 224), batch_size=64
)

class_names = train_ds.class_names
print("Обнаруженные классы:", class_names)

# Оптимизация загрузки данных
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Аугментация
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Создание базовой модели
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

# Замораживаем ВСЮ базовую модель
base_model.trainable = False

# Собираем окончательную модель
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.build(input_shape=(None, 224, 224, 3))
model.summary()

# Коллбэки для ранней остановки и адаптивного уменьшения LR
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)

# Обучение модели
history = model.fit(
    train_ds, validation_data=val_ds, epochs=5,  # Увеличили эпохи
    callbacks=[early_stopping, reduce_lr]
)

# Графики обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Точность на обучении')
plt.plot(epochs, val_acc, 'r', label='Точность на валидации')
plt.title('Точность обучения и валидации')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'b', label='Потери на обучении')
plt.plot(epochs, val_loss, 'r', label='Потери на валидации')
plt.title('Потери обучения и валидации')
plt.legend()
plt.show()

# Сохранение модели
model.save("trashnet_classifier.h5")
print("✅ Модель сохранена!")
