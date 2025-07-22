import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import os
from keras.models import load_model

st.header('Flower classification CNN Modele')
flower_name=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model=load_model('flower_recog_model.h5')

def classify_image(img):
    input_img = tf.keras.utils.load_img(img, target_size=(180, 180))
    input_img_array = tf.keras.utils.img_to_array(input_img)
    input_img_array = tf.keras.utils.img_to_array(input_img)
    input_img__exp_array = tf.expand_dims(input_img_array, 0)
    pre = model.predict(input_img__exp_array)
    result = tf.nn.softmax(pre[0])
    ans=flower_name[np.argmax(result)]
    return "The Image Belong to "+ ans + " category"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = os.path.join("upload", uploaded_file.name)

    # Ensure the 'upload' directory exists
    os.makedirs("upload", exist_ok=True)

    # Write the file to the directory
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, width=200)

    # Classify the uploaded image
    result = classify_image(temp_path)
    # Show classification result
    st.markdown(f"<h2 style='text-align: center; color: blue;'>{result}</h2>", unsafe_allow_html=True)
