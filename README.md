# 🍃 AI-Based Plant Disease Detection & Diagnosis System with XAI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-93.14%25-success.svg)

An advanced, lightweight Deep Learning application built to help farmers and agronomists identify plant diseases instantly. This system classifies **38 different plant diseases** across **14 crop species** using a highly optimized **MobileNetV2** architecture. 

To build trust and transparency in AI predictions, the project integrates **Explainable AI (XAI)** using **Grad-CAM heatmaps**, which visually highlight the exact infected regions on the leaf that led to the model's prediction.

---

## ✨ Key Features
*   **High Precision:** Achieves an overall accuracy of **93.14%** with high Macro F1-Scores evaluated under IEEE standard metrics.
*   **Explainable AI (Grad-CAM):** Generates real-time visual heatmaps overlaying the uploaded leaf image, showing the user *why* a specific disease was predicted.
*   **Confidence Guardrails:** The system incorporates a baseline confidence threshold (e.g., 50%). If the AI is unsure (due to blurry or non-leaf images), it safely rejects the input rather than making a false guess.
*   **Lightweight & Edge-Ready:** The entire model is just ~14 MB, making it highly suitable for deployment on mid-range smartphones and low-power IoT agricultural devices.
*   **Interactive Dashboard:** A clean, user-friendly frontend built with Streamlit for seamless image uploading and real-time inference.

---

## 🛠️ Technology Stack
*   **Deep Learning Framework:** TensorFlow & Keras
*   **Core Architecture:** MobileNetV2 (Pre-trained on ImageNet, Fine-tuned on PlantVillage dataset)
*   **Computer Vision:** OpenCV, NumPy
*   **Data Visualization & Reporting:** Matplotlib, Seaborn, Pandas, Scikit-learn
*   **Web Interface:** Streamlit

---

## 🚀 How to Setup and Run Locally

Follow these step-by-step instructions to run this project on your local machine.

### **1. Clone the Repository**
Open your terminal (or Command Prompt) and run:
`git clone https://github.com/Saksham-Chaudhary-03/Plant-Disease-Prediction.git`
`cd Plant-Disease-Prediction`

### **2. Create a Virtual Environment (Recommended)**
It is highly recommended to use a virtual environment to avoid library conflicts.
*   **For Windows:**
    `python -m venv env`
    `env\Scripts\activate`
*   **For macOS/Linux:**
    `python3 -m venv env`
    `source env/bin/activate`

### **3. Install Dependencies**
Install all required Python libraries using the `requirements.txt` file:
`pip install -r requirements.txt`

### **4. Run the Application**
Launch the Streamlit web application by executing:
`streamlit run app.py`
*A local web server will start, and the dashboard will open automatically in your default web browser.*

---

## 📊 Results & IEEE Standard Evaluation

The model has been rigorously evaluated using standard machine learning metrics. We dynamically calculate the following metrics based on actual model predictions:

*   **Accuracy:** 93.14%
*   **Macro Average Precision, Recall, and F1-Score:** Evaluated to ensure the model performs equally well across both majority and minority disease classes.
*   **Visual Artifacts:** The repository includes generated high-quality graphs such as:
    *   `Actual_Fig3_IEEE_Colored.png` (Comparison with baseline models)
    *   `Actual_Fig5_Top10_F1_Scores.png` (Detailed class-wise F1 score breakdown)

---

## 👥 Contributors

This project was developed collaboratively, combining expertise in Deep Learning architecture, model optimization, and frontend deployment.

*   **Saksham Chaudhary** 
    *   *Role:* Lead Developer & Repository Owner
    *   *Focus:* Model Architecture setup, Frontend UI (Streamlit) integration, System Workflow, and Version Control.
    *   *GitHub:* [@Saksham-Chaudhary-03](https://github.com/Saksham-Chaudhary-03)

*   **Harshita Tyagi**
    *   *Role:* Co-Developer & AI Researcher
    *   *Focus:* Model Optimization, Explainable AI (Grad-CAM) implementation, IEEE Metric Evaluations, and Algorithm Fine-tuning.

---

## 📜 License
This project is open-source and available under the **MIT License**.
