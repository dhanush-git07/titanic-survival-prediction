# Titanic Survival Prediction

**Author**: DHANUSH REDDY THUMMA
**Date**: May 2025  
**GitHub**: [github.com/dhanush-git07/titanic-survival-prediction](https://github.com/dhanush-git07/titanic-survival-prediction)

---

## 📋 Project Overview

- **Goal**: Build a model to predict whether a Titanic passenger survived.  
- **Dataset**: Kaggle “Titanic: Machine Learning from Disaster” (train.csv).  
- **Key Metric**: Accuracy on a held‐out test set.

---

## 🧮 Data & Features

1. **Raw Columns**  
   `PassengerId`, `Survived` (target), `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

2. **Data Cleaning**  
   - Filled missing **Age** with median.  
   - Filled missing **Embarked** with mode.  
   - Dropped **Cabin** (too many nulls).

3. **Original Engineered Features**  
   - `FamilySize = SibSp + Parch + 1`  
   - `IsAlone = 1 if FamilySize == 1 else 0`  
   - Encoded `Sex` → 0/1  
   - One-hot encoded `Embarked` → `Embarked_Q`, `Embarked_S`

4. **New Enriched Features**  
   - **Title** extracted from `Name` (Mr, Mrs, Miss, Master, Rare) → one-hot encoded  
   - **AgeGroup** (Child 0–12, Teen 12–18, Adult 18–35, Middle 35–60, Senior 60–120) → one-hot encoded  

---

## 🤖 Modeling

1. **Split 70/20/10** (train/validation/test), stratified by `Survived`.  

2. **Baseline Models (Original Features)**  
   - Logistic Regression → 82.02% validation accuracy  
   - Decision Tree → 83.15% validation accuracy  
   - Random Forest (default) → 83.71% validation accuracy  

3. **Hyperparameter Tuning on Original Features**  
   - GridSearchCV for Random Forest found best params but didn’t improve validation (still ~83.71%).

4. **Enriched Features + Tuning**  
   - Added Title & AgeGroup → one-hot encoded  
   - GridSearchCV found:  
     ```
     max_depth=5, max_features='sqrt',
     min_samples_leaf=1, min_samples_split=2,
     n_estimators=300
     ```  
   - **Validation accuracy**: 85.39%

---

## 🏁 Final Evaluation on Test Set

```python
from sklearn.metrics import accuracy_score, classification_report

y_test_pred = tuned_rf.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

- **Test Accuracy**: 0.8111

```text
precision    recall  f1-score   support
       0       0.83      0.94      0.88     100  
       1       0.84      0.63      0.72      50

accuracy                           0.81     150

---

## 📊 Interpretation & Insights

1. **Top-5 Feature Importances** (Random Forest):  
   1. Title_Mr  
   2. Sex  
   3. Fare  
   4. Pclass  
   5. Age  

2. **Error Analysis**  
   - False negatives often had titles like “Master” or “Miss” with moderate fare/class.  
   - False positives were frequently “Male” in 3rd class with low fare.

---

## 🚀 Demo & Deployment

**Live Streamlit App**  
Try the model online here:  
👉 [https://titanic-survival-prediction-65jqucqnunqwc5ipknnjeu.streamlit.app](https://titanic-survival-prediction-65jqucqnunqwc5ipknnjeu.streamlit.app)

 **Streamlit App** (`streamlit_app.py`):  
   - Enter passenger details (Pclass, Sex, Age, SibSp, Parch, Fare, etc.) in the sidebar.  
   - Click **Predict Survival** to view the survival probability and final prediction.
---

## 📚 Next Steps

- Explore additional feature interactions (e.g., `Sex × Pclass`).  
- Try other models (HistGradientBoosting, XGBoost) with enriched features.  
- Build a more robust API or dashboard for real-time predictions.

---

## 📄 How to Run This Project

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/ML-Project-Titanic.git
   cd ML-Project-Titanic
   ```

2. **Create & activate conda env**  
   ```bash
   conda env create -f environment.yml  # or pip install -r requirements.txt
   conda activate titanic-tf
   ```

3. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```  
   Open `notebooks/01_EDA_and_Preprocessing.ipynb`.

4. **Run final evaluation**  
   ```python
   from sklearn.metrics import accuracy_score, classification_report
   y_test_pred = tuned_rf.predict(X_test)
   print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
   print(classification_report(y_test, y_test_pred))
   ```

5. **Launch Streamlit Demo (Optional)**  
   ```bash
   streamlit run streamlit_app.py
   ```

---

_Happy modeling!_
