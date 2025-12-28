# 🧠 Suicide Detection Using Social Media Impressions

## 📌 Project Overview

This project focuses on **detecting suicidal intent from social media text data** using Natural Language Processing (NLP) and Machine Learning techniques.

The goal is to assist early identification of suicidal ideation by classifying user-generated text into:
- **Suicide**
- **Non-Suicide**
- **Borderline (Model Disagreement)**

The system uses a **dual-model approach**:
1. **Logistic Regression** with **TF-IDF features**
2. **BiLSTM (Bidirectional LSTM)** deep learning model

A **Streamlit web application** is provided to interactively test the models, visualize predictions, and maintain prediction history.

---

## 📊 Dataset

- **Source**: Kaggle  
- **Dataset Name**: Suicide and Depression Detection Dataset  
- **Link**:  
 https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data 

The dataset contains social media posts labeled as suicidal or non-suicidal and is used for training and evaluation.

---

## 🖥️ System Requirements

- **Python**: 3.10.5  
- **Operating System**: Windows / Linux / macOS  
- **Dependencies**: Listed in `requirements.txt`

> ⚠️ **Important**:  
> The models must be trained and loaded using the **same Python and library versions** to avoid compatibility issues.

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/stormbreaker3000/Suicide-Detection-Using-Social-Media-Impressions.git
cd Suicide-Detection-Using-Social-Media-Impressions
```

### 2️⃣ Create and Activate Virtual Environment

##### Windows
```
python -m venv venv
venv\Scripts\activate
```

##### Linux / macOS
```
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 4️⃣ 📂 Dataset Setup

1. Download the dataset from Kaggle  
   https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data

2. Create a folder named `Dataset` in the project root, if not present already.

3. Place the dataset files inside the `Dataset` folder.

Example structure:
```
project-root/
│
├── Dataset/
│ └── suicide_data.csv
```

### 5️⃣ 🧪 Model Training

1. Open the Jupyter Notebook:
```
jupyter notebook
```

2. Run the provided notebook step-by-step.
   - This will:
     - Clean and preprocess text
     - Train Logistic Regression (TF-IDF)
     - Train BiLSTM model
     - Save trained models into the `Models/` directory

After successful execution, the `Models` folder will be populated:

```
Models/
├── vectorizer.pkl
├── lr_model.pkl
├── tokenizer.pkl
└── bilstm_model.pkl
```
 
### 6️⃣ 🚀 Running the Streamlit App

Once models are generated, launch the web application:

```
streamlit run app.py
```

The app will be available at:
- **Local URL**: http://localhost:8501

---

## Project Structure
```
Suicide-Detection-Using-Social-Media-Impressions/
│
├── Models/ # Saved models (generated after notebook execution)
│ ├── Vectorizer_model.pkl
│ ├── LR_model.pkl
│ ├── Tokenizer_model.pkl
│ └── BILSTM_model.pkl
│
├── Dataset/ # Dataset files
│ └── suicide_data.csv
|
├── notebook.ipynb # Model training & evaluation
├── app.py # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🎯 Application Features

- Text input for sentiment analysis
- Predictions from **two independent models**
- **Final verdict** with disagreement handling
- Color-coded results:
  - 🟢 Non-Suicide
  - 🔴 Suicide
  - 🟡 Borderline
- Prediction history with visual indicators
- Graceful handling of missing models

---

## ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.  
It is **not a medical diagnostic tool** and should not replace professional mental health support.

If you or someone you know is struggling, please seek help from qualified professionals or local helplines.

---

## 👤 Author

**Subham Bagchi**  

