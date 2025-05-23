import streamlit as st
import pandas as pd
import joblib

# —————————————————————————————————————————————
#  Load the trained model and its expected columns
# —————————————————————————————————————————————
model = joblib.load('titanic_model.pkl')
model_cols = joblib.load('model_columns.pkl')

# —————————————————————————————————————————————
#  Page configuration (optional)
# —————————————————————————————————————————————
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# —————————————————————————————————————————————
#  Title and brief description
# —————————————————————————————————————————————
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
Enter the passenger’s details in the sidebar, then click **Predict Survival**.
This app uses a Random Forest model (trained on the Kaggle Titanic dataset)  
to estimate the probability of survival.
""")

# —————————————————————————————————————————————
#  Sidebar: Input widgets
# —————————————————————————————————————————————
with st.sidebar:
    st.header("Passenger Details")

    # Sex
    sex = st.selectbox(
        label="Sex",
        options=['male', 'female'],
        index=0
    )

    # Passenger Class
    pclass = st.selectbox(
        label="Passenger Class",
        options=[1, 2, 3],
        index=2,  # default to 3rd class (most common)
        help="1 = 1st class (highest), 3 = 3rd class (lowest)"
    )

    # Age (in years)
    age = st.number_input(
        label="Age (years)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.5,
        format="%.1f",
        help="Passenger age in years"
    )

    # Siblings/Spouses Aboard
    sibsp = st.number_input(
        label="Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of siblings or spouses aboard"
    )

    # Parents/Children Aboard
    parch = st.number_input(
        label="Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of parents or children aboard"
    )

    # Fare Paid (USD)
    fare = st.number_input(
        label="Fare Paid (USD)",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=0.5,
        format="%.2f",
        help="Ticket fare in US dollars"
    )

    # Embarked (binary flags for Q, S; C is implied if both unchecked)
    st.markdown("**Port of Embarkation**")
    embarked_s = st.checkbox("Southampton (S)", value=True)
    embarked_q = st.checkbox("Queenstown (Q)", value=False)
    # If neither S nor Q is checked, we assume Cherbourg (C) → both Embarked_S and Embarked_Q will be 0

    st.markdown("---")  # a horizontal separator
    # Predict button
    predict_button = st.button("Predict Survival")

# —————————————————————————————————————————————
#  Feature engineering (must match what model expects)
# —————————————————————————————————————————————
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
sex_num = 0 if sex == 'male' else 1

# Build the single‐row DataFrame in the same format as training
input_dict = {
    'Pclass': pclass,
    'Sex': sex_num,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Embarked_Q': int(embarked_q),
    'Embarked_S': int(embarked_s)
}

input_df = pd.DataFrame([input_dict])

# Reindex so columns match training order; any missing columns will be filled with 0
input_df = input_df.reindex(columns=model_cols, fill_value=0)

# —————————————————————————————————————————————
#  Prediction & display result
# —————————————————————————————————————————————
if predict_button:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Show the probability under the hood
    st.write(f"**Survival Probability:** {probability:.2%}")

    if prediction == 1:
        st.success("🎉 Predicted: Survived")
    else:
        st.error("☠️ Predicted: Did Not Survive")
