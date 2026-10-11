# Heart Disease Prediction System

An end-to-end Machine Learning project to predict the presence of heart disease based on clinical patient attributes. Built with Python, Scikit-Learn, and deployed as an interactive web application using Streamlit.

---

## 📌 Project Overview

- **Problem Type**: Binary Classification (Heart Disease: Present vs. Not Present)
- **Dataset**: Cleveland Heart Disease Database (UCI Machine Learning Repository)
- **Model**: Support Vector Classifier (`SVC` with RBF kernel and probability estimation)
- **Features Used**: 4 key clinical attributes:
  1. `Gender` (`Sex`): Male or Female
  2. `ChestPain` (`Chest Pain Type`): Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic
  3. `FastingBS` (`Fasting Blood Sugar`): > 120 mg/dl (True / False)
  4. `RestECG` (`Resting ECG`): Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy
- **Model Performance**: ~80% Accuracy on held-out test data.

---

## 📁 Repository Structure

```text
Heart disease/
├── .gitignore
├── requirements.txt                   # Project-level dependencies
├── runtime.txt                        # Python runtime for cloud deployment
├── README.md                          # Main project documentation
├── 1 - Heart Disease Prediction/      # Jupyter notebooks & training data
│   ├── Heart_Disease_Prediction.ipynb                  # Baseline modeling notebook
│   ├── Heart_Disease_Prediction_FullEnhancement.ipynb  # EDA & multi-model comparison
│   ├── heart_disease_data.csv                          # Cleaned dataset (4 features + target)
│   ├── svc_trained_model.pkl                           # Exported trained model artifact
│   └── requirements.txt                                # Notebook dependencies
├── heart+disease/                     # Original raw dataset from UCI
│   ├── processed.cleveland.data
│   └── heart-disease.names
└── streamlit-app/                     # Interactive Web Application
    ├── app.py                         # Streamlit application UI & prediction logic
    ├── svc_trained_model.pkl          # Trained model used by the web app
    ├── requirements.txt               # Streamlit deployment dependencies
    └── README.md                      # App-specific documentation
```

---

## 🚀 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Heart disease"
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
```bash
streamlit run streamlit-app/app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Jupyter Notebooks
```bash
jupyter notebook "1 - Heart Disease Prediction/Heart_Disease_Prediction_FullEnhancement.ipynb"
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select your GitHub repository.
4. Set **Main file path** to: `streamlit-app/app.py`
5. Click **"Deploy!"**.
