import os
import logging

# 1. Mute TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 2. Mute Python Engine Warnings
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Web Page Title and Description
st.title("🌿 Plant Disease Detector AI")
st.write("Upload one or multiple pictures of plant leaves to see if they are healthy or sick!")

# Load the model exactly once and cache it so the website stays fast
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_my_model()

# All 15 classes
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
    'Tomato_Tomato_YellowLeaf__Curl_Virus', 'Tomato_Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# Create the file uploader box (Allows multiple files!)
uploaded_files = st.file_uploader("Choose leaf images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# If the user uploads files, process them!
if uploaded_files:
    for file in uploaded_files:
        # 1. Open and display the image on the website
        image = Image.open(file)
        
        # We use columns to put the picture on the left, and the result on the right
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption=f"Uploaded: {file.name}", width=250)
            
        with col2:
            with st.spinner('AI is analyzing...'):
                # 2. Format the image for the AI (Resize to 256x256)
                image = image.convert('RGB')
                img = image.resize((256, 256))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                
                # 3. Make the prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = 100 * np.max(predictions[0])
                
                # 4. Display the results nicely
                st.subheader("Results:")
                st.write(f"**Diagnosis:** {predicted_class.replace('___', ' - ').replace('__', ' ')}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                if "healthy" in predicted_class.lower():
                    st.success("This leaf looks healthy! ✅")
                else:
                    st.error("Disease detected! ⚠️")
        
        # Add a dividing line between different uploaded images
        st.divider()