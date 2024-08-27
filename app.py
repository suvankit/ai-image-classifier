import streamlit as st
st.markdown('# Hello Aliens!')
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = 48

model = load_model("ai_generated_model.h5")

st.title("AI Image Classifier")       
        
img = st.file_uploader("Upload your Image")

if img and st.button("Check"):
    image = Image.open(img)
    st.image(img)
    image = ImageOps.fit(image, (48,48), Image.ANTIALIAS)
    img_array = img_to_array(image)
    new_arr = img_array/255
    test = []
    test.append(new_arr)
    test = np.array(test)
    y = model.predict(test)
    if y[0] <= 0.5:
        st.write("The given image is Real.")
    else:
        st.write("The given image is AI Generated.")
