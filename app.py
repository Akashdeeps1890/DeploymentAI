import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title('AI Solution Deployment - Final Exam')

st.header('Upload your data or use sample data')

# Upload data
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = load_iris(as_frame=True).frame

st.write(data.head())

# Split data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader('Analysis Results')
st.write(f'Accuracy: {accuracy:.2f}')

st.subheader('Predictions')
st.write(predictions)
