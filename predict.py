import os
import logging

# 1. Mute the C++ Engine (Must be BEFORE importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import tensorflow as tf
import numpy as np

# 2. Mute the Python Engine (Must be AFTER importing tensorflow)
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load your trained model
model = tf.keras.models.load_model("plant_disease_model.keras") 

# All 15 classes in perfect alphabetical order
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
    'Tomato_Tomato_YellowLeaf__Curl_Virus', 'Tomato_Tomato_mosaic_virus', 
    'Tomato_healthy'
]

def predict_image(image_path):
    # Load and format the image
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make the prediction (verbose=0 hides the loading bar)
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    
    # Translate the number into the plant name
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])
    
    print("\n" + "="*45)
    print(f"The model predicts: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*45 + "\n")

# Run it on your single test image!
IMAGE_PATH = "test_leaf1.jpg"
predict_image(IMAGE_PATH)