# ============================================================
# HEART DISEASE PREDICTION - VS CODE VERSION
# Dataset: HeartDiseaseTrain-Test.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    jaccard_score,
    roc_curve,
    auc
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "HeartDiseaseTrain-Test.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("HEART DISEASE PREDICTION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())


# ============================================================
# 2. BASIC EDA
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="target"
)

plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# Numerical correlation
plt.figure(figsize=(12, 8))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

TARGET = "target"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# Automatically identify categorical/numerical columns

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

print("\nNumerical Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_cols
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_cols
    )
])


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ============================================================
# 6. DEFINE ML MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=7
        )
}


# ============================================================
# 7. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []
pipelines = {}


# ============================================================
# 8. TRAIN ALL MODELS
# ============================================================

for name, model in models.items():

    print("\n")
    print("=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipelines[name] = pipeline

    # Cross Validation

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    print("\n5-Fold CV Scores:")
    print(cv_scores)

    print(
        "Mean CV Accuracy:",
        round(cv_scores.mean(), 4)
    )

    # Train

    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # METRICS

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    logloss = log_loss(
        y_test,
        y_prob
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    hamming = hamming_loss(
        y_test,
        y_pred
    )

    jaccard = jaccard_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # PRINT ALL METRICS

    print("\n--------------- ALL METRICS ---------------")

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy   : {balanced_acc:.4f}"
    )

    print(
        f"Precision           : {precision:.4f}"
    )

    print(
        f"Recall              : {recall:.4f}"
    )

    print(
        f"F1 Score            : {f1:.4f}"
    )

    print(
        f"ROC-AUC             : {roc_auc:.4f}"
    )

    print(
        f"Log Loss            : {logloss:.4f}"
    )

    print(
        f"Matthews Corr.      : {mcc:.4f}"
    )

    print(
        f"Cohen Kappa         : {kappa:.4f}"
    )

    print(
        f"Hamming Loss        : {hamming:.4f}"
    )

    print(
        f"Jaccard Score       : {jaccard:.4f}"
    )


    # CONFUSION MATRIX

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)


    # CLASSIFICATION REPORT

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    # SAVE RESULTS

    results.append({

        "Model": name,

        "CV Accuracy":
            cv_scores.mean(),

        "Accuracy":
            accuracy,

        "Balanced Accuracy":
            balanced_acc,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc,

        "Log Loss":
            logloss,

        "MCC":
            mcc,

        "Cohen Kappa":
            kappa,

        "Hamming Loss":
            hamming,

        "Jaccard":
            jaccard
    })


# 9. MODEL COMPARISON

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "CV Accuracy",
        "ROC-AUC"
    ],
    ascending=False
)

print("\n\n")
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# 10. BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

best_model = pipelines[
    best_model_name
]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Model:",
    best_model_name
)

# 11. FINAL EVALUATION OF BEST MODEL

best_pred = best_model.predict(
    X_test
)

best_prob = best_model.predict_proba(
    X_test
)[:, 1]


print("\nFinal Classification Report:\n")

print(
    classification_report(
        y_test,
        best_pred,
        digits=4
    )
)

# 12. FINAL CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    best_pred
)

print("\nFinal Confusion Matrix:")
print(cm)


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Disease",
        "Disease"
    ],
    yticklabels=[
        "No Disease",
        "Disease"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# 13. ROC CURVE

fpr, tpr, thresholds = roc_curve(
    y_test,
    best_prob
)

roc_auc_value = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC = {roc_auc_value:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# 14. FINAL METRICS

print("\n")
print("=" * 70)
print("FINAL BEST MODEL METRICS")
print("=" * 70)

print(
    f"Accuracy          : {accuracy_score(y_test, best_pred):.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy_score(y_test, best_pred):.4f}"
)

print(
    f"Precision         : {precision_score(y_test, best_pred):.4f}"
)

print(
    f"Recall            : {recall_score(y_test, best_pred):.4f}"
)

print(
    f"F1 Score          : {f1_score(y_test, best_pred):.4f}"
)

print(
    f"ROC-AUC           : {roc_auc_score(y_test, best_prob):.4f}"
)

print(
    f"Log Loss          : {log_loss(y_test, best_prob):.4f}"
)

print(
    f"MCC               : {matthews_corrcoef(y_test, best_pred):.4f}"
)

print(
    f"Cohen Kappa       : {cohen_kappa_score(y_test, best_pred):.4f}"
)

print(
    f"Hamming Loss      : {hamming_loss(y_test, best_pred):.4f}"
)

print(
    f"Jaccard Score     : {jaccard_score(y_test, best_pred):.4f}"
)

# 15. SAVE BEST MODEL

MODEL_PATH = "heart_disease_best_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved as:")
print(MODEL_PATH)

# 16. SAMPLE PREDICTION

sample = X_test.iloc[
    [0]
]

sample_prediction = best_model.predict(
    sample
)[0]

sample_probability = best_model.predict_proba(
    sample
)[0][1]


print("\n")
print("=" * 70)
print("SAMPLE PATIENT PREDICTION")
print("=" * 70)

print("\nPatient Data:")
print(sample.to_string(index=False))

print(
    "\nPredicted Class:",
    sample_prediction
)

print(
    f"Probability of Heart Disease Class: "
    f"{sample_probability:.2%}"
)


if sample_prediction == 1:

    print(
        "\nPrediction: Heart disease class"
    )

else:

    print(
        "\nPrediction: No heart disease class"
    )


print("\n")
print("=" * 70)
print("PROGRAM COMPLETED")
print("=" * 70)