import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

# Загружаем базу признаков
conv_base = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))

# Загружаем обученную модель
model = keras.models.load_model("clothes_classifier.keras")

# Функция предсказания
def predict_image(image_path):
    img = load_img(image_path, target_size=(180, 180))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.vgg16.preprocess_input(img_array)

    features = conv_base.predict(img_array)
    prediction = model.predict(features)
    class_index = np.argmax(prediction, axis=1)[0]

    class_names = ["jacket", "poloshirt", "shirt", "tanktop", "tshirt", "warmclothes"]
    return class_names[class_index]

# Пример использования:
print(predict_image("train_clo/jacket.410.jpg"))