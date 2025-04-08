# College Chatbot

![Chatbot Screenshot](<PASTE_YOUR_IMAGE_LINK_HERE>)

---

## Tools and Technologies Used

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-lightgrey)
![NLTK](https://img.shields.io/badge/NLTK-Natural_Language_Toolkit-green)
![HTML](https://img.shields.io/badge/HTML-5-orange)
![CSS](https://img.shields.io/badge/CSS-Professional_Styling-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-Client_Side-yellow)

---

## 📘 Project Overview

This project is an AI-powered rule-based **College Chatbot** that simulates answering student queries related to college facilities, admission, contact details, fees, and more. It has a web-based user interface created using **Flask** and **HTML/CSS**, backed by a simple machine learning model for intent classification.

---

## 🛠️ Step-by-Step Workflow

### 1. Data Preparation
- Created a custom `intents.json` file containing intents, sample queries, and expected responses.
- Preprocessed the data using **NLTK** for tokenization and stemming.

### 2. Model Development
- Used **TensorFlow (Keras)** to build a neural network classifier.
- The model was trained to classify user queries into predefined intents.
- Saved trained model and vocabulary using `model.h5`, `vocab.pkl`, and `label_encoder.pkl`.

### 3. Chat Functionality
- Developed a `chatbot.py` script to process user input and predict appropriate responses using the trained model.

### 4. Web Integration
- Built a Flask backend to handle user requests from the browser.
- Designed a professional front-end using **HTML, CSS, and JavaScript**.
- Handled real-time chat communication using asynchronous JavaScript with fetch API.

### 5. Testing and Deployment
- Successfully tested various user inputs and intent responses.
- Ensured the chatbot could serve information accurately and quickly through the web interface.

---

## 👤 Author

**Manas Bhise**  
*Project Created: April 2025*
