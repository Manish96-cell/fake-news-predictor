import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models
model = BertForSequenceClassification.from_pretrained("models/misinformation_model")
tokenizer = BertTokenizer.from_pretrained("models/misinformation_model")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
API_KEY = 'AIzaSyC8SxCy92vwF0gpjZ2RF7uyolVcKaUJjMc'
service = build('kgsearch', 'v1', developerKey=API_KEY)

# Define the misinformation detection function
def predict_misinformation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction

# Google Knowledge Graph Fact-Checking
def get_google_kg_entity(query):
    response = service.entities().search(query=query, limit=1).execute()
    if 'itemListElement' in response:
        entity = response['itemListElement'][0]
        entity_name = entity['result']['name']
        entity_description = entity['result']['description'] if 'description' in entity['result'] else "No description available"
        entity_url = entity['result']['url'] if 'url' in entity['result'] else None
        return entity_name, entity_description, entity_url
    else:
        return None, None, None

def compute_similarity(input_text, page_content):
    input_embedding = embedder.encode([input_text])
    page_embedding = embedder.encode([page_content])
    return cosine_similarity(input_embedding, page_embedding)[0][0]

def user_friendly_fact_check(input_text, threshold=0.75):
    key_terms = input_text.split()[:3]
    best_match_score = 0
    best_match_details = None

    for term in key_terms:
        name, description, url = get_google_kg_entity(term)
        if name:
            similarity_score = compute_similarity(input_text, description)
            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match_details = {"name": name, "description": description, "url": url}

    if best_match_score > threshold:
        result = "✅ Fact Check Passed"
        confidence = "High Confidence"
        recommendation = "No further verification needed."
    elif best_match_score > 2*threshold/3:
        result = "⚠️ Likely True"
        confidence = "Moderate Confidence"
        recommendation = "Verify further using reliable sources."
    else:
        result = "❌ Likely False"
        confidence = "Low Confidence"
        recommendation = "Check with multiple sources for accuracy."

    return {
        "Fact-Check Result": result,
        "Entity Name": best_match_details['name'] if best_match_details else "No match found",
        "Description": best_match_details['description'][:300] + "..." if best_match_details else "No description available",
        "Confidence Level": confidence,
        "Similarity Score": round(best_match_score, 2),
        "Next Step": recommendation,
        "Entity URL": best_match_details['url'] if best_match_details else "No URL available"
    }

# Streamlit App UI
st.set_page_config(page_title="🔍Misinformation Detection", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #d0e1f9;
        text-align: center;
    }
    .sub-title {
        font-size: 18px;
        color: #6c757d;
        text-align: center;
    }
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🕵️Misinformation Detection and Fact-Checking🕵️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze text and ensure its authenticity</div>', unsafe_allow_html=True)

# Sidebar for advanced options
st.sidebar.markdown("### Advanced Options")
threshold = st.sidebar.slider("Set Similarity Threshold", 0.0, 1.0, 0.75)

# Main input and processing
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    user_input = st.text_area("Enter text to check", placeholder="Type or paste the text you want to analyze here...", height=200)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Check News"):
        if user_input:
            with st.spinner("Analyzing news..."):
                prediction = predict_misinformation(user_input)
                if prediction == 1:
                    st.success("This news is real.")
                else:
                    st.error("This news is fake.")
        else:
            st.warning("Please enter some text.")

with col_b:
    if st.button("Check Facts"):
        if user_input:
            with st.spinner("Fact-checking..."):
                result = user_friendly_fact_check(user_input, threshold)
                if result['Fact-Check Result'] == "✅ Fact Check Passed":
                    st.success(f"Fact-Check Result: {result['Fact-Check Result']}")
                elif result['Fact-Check Result'] == "⚠️ Likely True":
                    st.warning(f"Fact-Check Result: {result['Fact-Check Result']}")
                else:
                    st.error(f"Fact-Check Result: {result['Fact-Check Result']}")
                st.markdown(f"**Confidence Level:** {result['Confidence Level']}")
                st.markdown(f"**Similarity Score:** {result['Similarity Score']}")
                if result['Entity URL'] != "No URL available":
                    st.markdown(f"[More Info]({result['Entity URL']})")
        else:
            st.warning("Please enter some text.")

# Expandable help section
with st.expander("ℹ️ How to Use"):
    st.write("""
        - Enter the text you want to verify.
        - Click the appropriate button to check for misinformation or fact-check.
    """)
