# 🍎 Fruit Freshness Detector using Deep Learning

A Deep Learning web application that classifies fruit images as **Fresh** 🍏 or **Rotten** 🍂 using a Convolutional Neural Network (CNN). The application is built with **TensorFlow/Keras** and deployed using **Streamlit** for real-time predictions.

---

## 🚀 Live Demo

🌐 **Live Application:** *https://fruit-freshness-detector-vxaxobt4gvictltjafm4u7.streamlit.app/*

💻 **GitHub Repository:** *https://github.com/Shashwatsoni06/Fruit-Freshness-Detector*

---

## 📌 Project Overview

Fruit quality inspection is an important task in agriculture, food processing, and retail industries. Manual inspection is time-consuming and prone to human error.

This project uses **Computer Vision** and **Deep Learning** to automatically classify fruit images into:

* 🍏 Fresh Fruit
* 🍂 Rotten Fruit

Users can upload an image through the web interface, and the model predicts whether the fruit is fresh or rotten, along with the confidence score.

---

## ✨ Features

* Upload fruit images (JPG, JPEG, PNG)
* Automatic image preprocessing
* CNN-based image classification
* Fresh vs Rotten prediction
* Confidence score
* Probability visualization
* Interactive and responsive Streamlit UI
* Deployed online using Streamlit Community Cloud

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pillow**
* **Plotly**
* **Streamlit**

---

## 🧠 Model Architecture

The model is built using a **Convolutional Neural Network (CNN)** consisting of:

* Convolution Layers
* Max Pooling Layers
* Batch Normalization
* Dropout
* Dense Layers
* Sigmoid Activation for Binary Classification

The model is trained using:

* Binary Cross-Entropy Loss
* Adam Optimizer

---

## 📂 Project Structure

```text
Fruit-Freshness-Detector/
│
├── app.py
├── fruits_classification_model.keras
├── requirements.txt
├── README.md
├── assets/
└── runtime.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Fruit-Freshness-Detector.git
```

Move into the project folder:

```bash
cd Fruit-Freshness-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📷 How It Works

1. Upload a fruit image.
2. The image is resized to **224 × 224**.
3. Pixel values are normalized.
4. The trained CNN model predicts the class.
5. The application displays:

   * Prediction
   * Confidence Score
   * Fresh Probability
   * Rotten Probability

---

## 📊 Classes

| Label | Description     |
| ----- | --------------- |
| 0     | Fresh Fruit 🍏  |
| 1     | Rotten Fruit 🍂 |

---

## 🎯 Future Improvements

* Support multiple fruit categories
* Transfer Learning using EfficientNet or MobileNet
* Batch image prediction
* Explainable AI (Grad-CAM visualization)
* Mobile-friendly interface
* Docker deployment
* REST API integration

---

## 📸 Screenshots
<img width="2878" height="1420" alt="Screenshot 2026-06-25 140151" src="https://github.com/user-attachments/assets/638b131e-d834-43bf-9d9f-016dfc8421d2" />
<img width="2880" height="1354" alt="Screenshot 2026-06-25 140256" src="https://github.com/user-attachments/assets/d0aa66c3-a6d1-4a47-86a4-f19b59366fd0" />




## 🤝 Connect With Me

**Shashwat Soni**

📧 Email: *shashwatsoni06@gmail.com*

💼 LinkedIn: *www.linkedin.com/in/shashwat-soni-b3a1ba233*

🐙 GitHub: https://github.com/Shashwatsoni06

---

## ⭐ If you found this project interesting, consider giving it a star on GitHub!
