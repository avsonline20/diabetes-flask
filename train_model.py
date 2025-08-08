# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 1. load
df = pd.read_csv('diabetes.csv')  

# 2. features and label
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. handle impossible zeros for some cols
cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
X[cols_with_zero] = X[cols_with_zero].replace(0, np.nan)

# 4. pipeline: imputer, scaler, model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
if hasattr(pipeline, "predict_proba"):
    print('ROC AUC:', roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))

# 6. save model
joblib.dump(pipeline, 'model.joblib')
print('Saved model.joblib')
