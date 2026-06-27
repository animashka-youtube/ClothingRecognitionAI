import streamlit as st
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
# добавить подпись
st.title("Классификатор одежды")

model = keras.models.load_model("clothes_classifier.keras")
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))
class_names = ["Жакет", "Рубашка поло", "рубашка", "Майка", "футболка", "Тёплая одежда"]

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    img = image.resize((180, 180))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = conv_base.predict(img_array)
    prediction = model.predict(features)
    class_index = np.argmax(prediction, axis=1)[0]

    st.success(f"Это: **{class_names[class_index]}**")

# streamlit run interface_better.py. Остановка Ctrl + C