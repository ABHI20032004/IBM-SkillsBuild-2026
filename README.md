# AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares
This is the final project I submitted as the final project for evaluation of what i did learn during the internship period.


❤️ Heart Disease Prediction using Machine Learning

A machine learning classification project that predicts the heart disease target class from clinical and patient-related features.

Note: This project is for educational/academic purposes. It is not a medical diagnostic system and should not be used to make clinical decisions.

📌 Project Overview

This project uses the HeartDiseaseTrain-Test.csv dataset and compares multiple machine learning classification algorithms.


📊 Dataset link : https://www.kaggle.com/datasets/yangfanc/heart-disease-dataset-uci


🤖 Models Trained

The following models were trained and compared:

Logistic Regression

Random Forest

Extra Trees

Gradient Boosting

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

A 5-fold Stratified Cross Validation was used during model comparison.


🔲 Confusion Matrix

The test set contained:

100 actual No Disease samples

105 actual Disease samples


📉 ROC Curve

The Extra Trees model produced:

ROC-AUC = 1.0000


📊 Target Distribution

The target distribution contains:

Class 0 → 499 samples
Class 1 → 526 samples



🧪 Example Prediction

A sample test patient was passed through the trained Extra Trees model.

The output was:

Predicted Class: 0
Probability of Heart Disease Class: 0.00%

Prediction: No heart disease class

This prediction is an output of the trained ML model and should not be interpreted as a clinical diagnosis.

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

📁 Project Structure

Heart-Disease-Prediction/
│
├── HeartDiseaseTrain-Test.csv
├── Abhishek_Raj__Heart_Disease_Prediction.py
├── requirements.txt
├── heart_disease_best_model.pkl
├── README.md
│
└── images/
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── correlation_heatmap.png
    └── target_distribution.png

▶️ How to Run

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Heart-Disease-Prediction

2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the project

python Abhishek_Raj__Heart_Disease_Prediction.py

📦 Model Output

After training, the best model is saved as:

heart_disease_best_model.pkl


👨‍💻 Author

Abhishek Raj

B.Tech — Artificial Intelligence and Data Engineering

📄 Project Purpose

This project demonstrates a complete classical machine learning workflow:

Data → EDA → Preprocessing → Model Training → Cross Validation → Evaluation → Model Selection → Prediction