import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("models/misinformation_model")
tokenizer = BertTokenizer.from_pretrained("models/misinformation_model")

# Define function to make predictions
def predict_misinformation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction

# Streamlit app UI
st.title("Misinformation Detection")
st.write("Enter news content to check if it's real or fake.")

# Text input for user
user_input = st.text_area("News Text", height=200)

# Button to make prediction
if st.button("Predict"):
    if user_input:
        prediction = predict_misinformation(user_input)
        if prediction == 1:
            st.success("This news is real.")
        else:
            st.error("This news is fake.")
    else:
        st.warning("Please enter some text.")
