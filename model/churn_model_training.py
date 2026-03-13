import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
SEED = 42
np.random.seed(SEED)

df = pd.read_excel("data/Churn_Modelling.xlsx")

print("Dataset Shape:", df.shape)
print(df.head())

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# EDA
print("\nDataset Info\n")
print(df.info())

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nSummary Statistics\n")
print(df.describe())

# Encoding Categorical Variables

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])

# Correlation Analysis
corr_mat = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_mat, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Correlation with Target Variable
target_corr = corr_mat["Exited"].abs().sort_values(ascending=False)

print("\nCorrelation with Target Variable:\n")
print(target_corr)


# Selecting Top 7 Features
top_feat = target_corr.drop("Exited").head(7).index.tolist()

print("\nTop 7 Features:\n", top_feat)


# Defining Features and Label

X = df[top_feat]
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)


#Cross Validation
cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\nCross Validation Scores:", cv_scores)
print("Average CV Accuracy:", np.mean(cv_scores))

#Predicting Probabilities
y_probs = rf_model.predict_proba(X_test)[:,1]

#Finding the  Best Threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

f1 = (2 * precision * recall) / (precision + recall + 1e-6)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print("\nBest Threshold:", best_threshold)
plt.figure(figsize=(8,5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold")
plt.legend()
plt.show()

y_pred = (y_probs >= best_threshold).astype(int)

#Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# 18. Feature Importance
importance = pd.Series(
    rf_model.feature_importances_,
    index=top_feat
).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()
df["Churn_Probability"] = rf_model.predict_proba(X)[:,1]

df.to_csv("customer_churn_scores.csv", index=False)

