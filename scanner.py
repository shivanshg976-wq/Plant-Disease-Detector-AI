import os
import logging

# 1. Mute the C++ Engine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import tensorflow as tf
import numpy as np

# 2. Mute the Python Engine
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("Waking up the AI...")
model = tf.keras.models.load_model("plant_disease_model.keras") 

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

def scan_folder(folder_path):
    print("\n" + "="*55)
    print(f"Scanning all images in folder: '{folder_path}'")
    print("="*55)
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Could not find a folder named '{folder_path}'")
        print("Please create it and add some images!")
        return

    # Loop through every single file in the folder
    files_scanned = 0
    for filename in os.listdir(folder_path):
        # Only check image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            files_scanned += 1
            image_path = os.path.join(folder_path, filename)
            
            # Prepare the image
            img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Make the prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = 100 * np.max(predictions[0])
            
            # Print the result for this specific file
            print(f"📄 {filename.ljust(20)} --> 🌿 {predicted_class} ({confidence:.1f}%)")
            
    if files_scanned == 0:
        print("Folder is empty! No images found to scan.")

# Run the scanner on your folder!
FOLDER_NAME = "test_batch"
scan_folder(FOLDER_NAME)
print("\nScan Complete!\n")