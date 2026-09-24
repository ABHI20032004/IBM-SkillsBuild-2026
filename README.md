# AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares
This is the final project I submitted as the final project for evaluation of what i did learn during the internship period.


❤️ Heart Disease Prediction using Machine Learning

A machine learning classification project that predicts the heart disease target class from clinical and patient-related features.

Note: This project is for educational/academic purposes. It is not a medical diagnostic system and should not be used to make clinical decisions.

📌 Project Overview

This project uses the HeartDiseaseTrain-Test.csv dataset and compares multiple machine learning classification algorithms.

The complete pipeline includes:

Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Train / Test Split
   ↓
Multiple ML Models
   ↓
5-Fold Cross Validation
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction

📊 Dataset

The dataset contains:

1,025 samples

14 columns

13 input features

1 target column

There were no missing values in the dataset.

Numerical Features

age

resting_blood_pressure

cholestoral

Max_heart_rate

oldpeak

Categorical Features

sex

chest_pain_type

fasting_blood_sugar

rest_ecg

exercise_induced_angina

slope

vessels_colored_by_flourosopy

thalassemia

Target

0 → No Disease
1 → Disease

The dataset contains:

Target

Samples

0

499

1

526

🤖 Models Trained

The following models were trained and compared:

Logistic Regression

Random Forest

Extra Trees

Gradient Boosting

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

A 5-fold Stratified Cross Validation was used during model comparison.

📈 Model Results

The obtained results were:

Model

CV Accuracy

Test Accuracy

Precision

Recall

F1

ROC-AUC

Extra Trees

98.41%

100.00%

100.00%

100.00%

100.00%

100.00%

Random Forest

98.29%

100.00%

100.00%

100.00%

100.00%

100.00%

Gradient Boosting

96.34%

99.02%

98.13%

100.00%

99.06%

99.31%

SVM

90.24%

95.12%

93.58%

97.14%

95.33%

97.73%

Logistic Regression

86.83%

87.32%

85.59%

90.48%

87.96%

94.45%

KNN

85.73%

88.78%

87.27%

91.43%

89.30%

96.31%

The model-selection procedure selected:

Best Model: Extra Trees

The Extra Trees model achieved a 5-fold cross-validation accuracy of 98.41% on the training portion and 100% accuracy on the held-out test set.

Because perfect test performance is unusually high for a medical prediction dataset, this result should be investigated for possible dataset duplication, leakage, or other characteristics before treating it as evidence of real-world performance.

📋 Detailed Best Model Metrics

Extra Trees

Accuracy          : 1.0000
Balanced Accuracy : 1.0000
Precision         : 1.0000
Recall            : 1.0000
F1 Score          : 1.0000
ROC-AUC           : 1.0000
Log Loss          : 0.0008
MCC               : 1.0000
Cohen Kappa       : 1.0000
Hamming Loss      : 0.0000
Jaccard Score     : 1.0000

🔲 Confusion Matrix

The test set contained:

100 actual No Disease samples

105 actual Disease samples

The Extra Trees confusion matrix was:

                 Predicted
              No Disease  Disease
Actual
No Disease        100        0
Disease             0      105



📉 ROC Curve

The Extra Trees model produced:

ROC-AUC = 1.0000



🔥 Correlation Heatmap

The correlation heatmap was generated for the numerical variables in the dataset.



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

The saved pipeline contains the preprocessing and trained model so that it can be reused for prediction.

⚠️ Important Considerations

The reported results are based on this particular dataset and train/test split.

The 100% test performance should not be interpreted as 100% real-world accuracy. Before deployment or clinical use, the model would require:

External validation

Evaluation on an independent dataset

Investigation of duplicate or highly similar records

Leakage analysis

Calibration analysis

Clinical validation

Appropriate privacy and regulatory review

👨‍💻 Author

Abhishek Raj

B.Tech — Artificial Intelligence and Data Engineering

📄 Project Purpose

This project demonstrates a complete classical machine learning workflow:

Data → EDA → Preprocessing → Model Training → Cross Validation → Evaluation → Model Selection → Prediction