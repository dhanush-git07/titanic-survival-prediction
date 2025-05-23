# Titanic Survival Prediction

**Author**: _Your Name_  
**Date**: May 2025  
**GitHub**: [github.com/yourusername/ML-Project-Titanic](https://github.com/yourusername/ML-Project-Titanic)

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

- **Test Accuracy**: _Your final number here (e.g., 0.85xx)_  

---

## 📊 Interpretation & Insights

1. **Top-5 Feature Importances** (Random Forest):  
   1. Sex  
   2. Pclass  
   3. Title_Miss  
   4. AgeGroup_Adult  
   5. Fare  

2. **Error Analysis**  
   - False negatives often had titles like “Master” or “Miss” with moderate fare/class.  
   - False positives were frequently “Male” in 3rd class with low fare.

---

## 🚀 Demo & Deployment (Optional)

1. **Streamlit App** (`streamlit_app.py`):  
   - UI to enter passenger features → shows survival probability.  
2. **Flask API** (`app.py`):  
   - `/predict` endpoint that returns JSON `{ "survived": 0/1, "probability": 0.85 }`.  
3. **Dockerfile** Example:  
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

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
