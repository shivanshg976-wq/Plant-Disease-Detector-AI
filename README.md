# 🌿 Plant Disease Detector AI

## 📌 Description

Plant Disease Detector AI is a machine learning-based system that uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images. The system can classify 15 different conditions across Pepper, Potato, and Tomato plants.

It supports both **command-line usage** and an optional **Streamlit web interface**, making it flexible for developers, researchers, and farmers.

---

## ❗ Problem Statement

Plant diseases significantly impact agricultural productivity. Early detection is crucial but often requires expert knowledge. Many farmers lack access to such expertise.

This project aims to:

* Automate disease detection
* Provide quick and accurate diagnosis
* Reduce dependency on manual inspection

---

## 🚀 Features

* ✅ Detects 15 plant disease classes
* ✅ Supports Pepper, Potato, and Tomato plants
* ✅ CLI-based prediction for single image
* ✅ Bulk image scanning using folder input
* ✅ Streamlit web interface (optional)
* ✅ Displays disease name with confidence %
* ✅ Custom CNN trained from scratch
* ✅ Data augmentation for improved accuracy

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Machine Learning:** TensorFlow, Keras
* **Data Processing:** NumPy
* **Image Processing:** PIL (Pillow)
* **Visualization:** Matplotlib
* **Web Interface (Optional):** Streamlit

---

## 📂 Project Structure

```
Plant-Disease-Detector-AI/
│
├── dataset/                  # Training dataset (PlantVillage subset)
├── train_model.py            # Model training script
├── predict.py                # Predict single image (CLI)
├── scanner.py                # Scan folder of images (CLI)
├── app.py                    # Streamlit web app
├── plant_disease_model.keras # Trained model file
├── class_names.json          # Class labels
├── test_batch/               # Folder for bulk testing
├── test_leaf1.jpg            # Sample test image
└── README.md
```

---

## ⚙️ Installation Instructions

### 1. Clone the Repository

```bash
git clone(https://github.com/shivanshg976-wq/Plant-Disease-Detector-AI)
cd plant-disease-detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Linux/Mac:

```bash
source venv/bin/activate
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install tensorflow numpy pillow matplotlib streamlit
```

---

## ▶️ How to Run the Project

### 1. Train the Model

```bash
python train_model.py
```

---

### 2. Predict Single Image

```bash
python predict.py
```

(Change image path inside the file if needed)

---

### 3. Scan Folder of Images

```bash
python scanner.py
```

(Add images to `test_batch/` folder)

---

### 4. Run Web App (Optional)

```bash
streamlit run app.py
```

---

## 📖 Usage Instructions

### Single Image Prediction

* Place your image in project folder
* Update path in `predict.py`
* Run script

### Bulk Prediction

* Add images to `test_batch/`
* Run `scanner.py`

### Web Interface

* Upload one or multiple images
* View results instantly

---

## 🖼️ Sample Output / Screenshots
<img width="1555" height="605" alt="Screenshot 2026-03-27 042125" src="https://github.com/user-attachments/assets/8b3b8537-34bb-4138-a523-ef41deb2634b" />
<img width="954" height="135" alt="Screenshot 2026-03-27 172003" src="https://github.com/user-attachments/assets/604836d1-e5c5-4d35-a086-141c7620a4a7" />
<img width="1881" height="938" alt="image" src="https://github.com/user-attachments/assets/5c81c6b1-e29a-4465-b952-10dadc43540d" />

>

Example:

* Training accuracy graph
* Prediction output
* Web interface UI

---

## 🔮 Future Improvements

* Support more crop types
* Mobile application integration
* Real-time camera detection
* Cloud deployment (AWS/GCP)
* Improved accuracy using transfer learning

---

## 🤝 Contribution Guidelines

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit and push
5. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Shivansh Gupta
25BCE10924
