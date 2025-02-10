# 🌟Building Classification API

A FastAPI-based machine learning model that classifies building types (**Bungalow, High-rise, Storey-Building**) from images using a **Convolutional Neural Network (CNN)**.

---

## 🔄 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [License](#-license)

---

## 📚 Introduction

This project is a **deep learning-based image classification API** that identifies different building types using a **CNN model trained with PyTorch**. It is built with **FastAPI** for deployment and supports **image uploads** via HTTP requests.

---

## ✨ Features

✅ Classifies buildings into **Bungalow, High-rise, and Storey-Building**\
✅ Built with **FastAPI** for high performance\
✅ Uses **PyTorch and Torchvision** for deep learning inference\
✅ **Optimized with GPU (CUDA) support** for fast processing

---

## 🛠 Technologies Used

- Python 3.8+
- PyTorch (for deep learning)
- FastAPI (for web API)
- Pillow (for image processing)
- Uvicorn (for running the server)

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yusufab/building-classification-api.git
cd building-classification-api
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the trained model

Ensure you have the model file `cnn_model.pth` in the root directory.

---

## 🚀 Usage

### Run the API server

```bash
uvicorn main:app --reload
```

The API will be accessible at:\
👉 `http://127.0.0.1:8000/docs` *(Swagger UI for testing the API)*

---

## 📊 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input size:** `400x300` (RGB images)
- **Output classes:** Bungalow, High-rise, Storey-Building
- **Normalization:** `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`
- **Optimization:** Dropout (0.4), Batch Normalization

---

## 📚 License

This project is licensed under the MIT License.

---


