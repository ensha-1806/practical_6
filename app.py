import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("Titanic Survival Prediction App")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Titanic Dataset CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # ---------------------------
    # Data Preprocessing
    # ---------------------------
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
            'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']

    # ---------------------------
    # Train Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---------------------------
    # Show Accuracy
    # ---------------------------
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Model Accuracy")
    st.write(round(acc, 3))

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------------------
    # Manual Prediction Section
    # ---------------------------
    st.subheader("Predict Survival")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.number_input("Age", min_value=0)
    sibsp = st.number_input("Siblings/Spouses", min_value=0)
    parch = st.number_input("Parents/Children", min_value=0)
    fare = st.number_input("Fare", min_value=0.0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    embarked = st.selectbox("Embarked", ["Q", "S"])

    if st.button("Predict"):

        sex_male = 1 if sex == "Male" else 0
        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0

        input_data = np.array([[pclass, age, sibsp, parch, fare,
                                sex_male, embarked_Q, embarked_S]])

        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("Passenger Survived ✅")
        else:
            st.error("Passenger Did Not Survive ❌")
