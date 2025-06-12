import streamlit as st
import re
import joblib

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Extract dataset mentions from input text
def extract_dataset_mentions(text):
    doi_pattern = re.compile(r'(10\.\d{4,9}/[^\s";]+)', re.IGNORECASE)
    acc_pattern = re.compile(r'\b(GSE\d+|E-\w+-\d+|PRJ\w+\d+|CHEMBL\d+|PDB\s+\w+)\b', re.IGNORECASE)
    matches = set(doi_pattern.findall(text)) | set(acc_pattern.findall(text))
    return matches

# Streamlit app
st.title("Dataset Citation Classifier (Primary or Secondary?)")
st.markdown("Paste a paragraph from a scientific paper. The app will find dataset mentions and classify them.")

user_input = st.text_area("Enter paragraph:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please paste some text.")
    else:
        mentions = extract_dataset_mentions(user_input)
        if not mentions:
            st.info("No dataset mentions found.")
        else:
            st.subheader("Predictions")
            for mention in mentions:
                context = mention
                X = vectorizer.transform([context])
                pred = model.predict(X)[0]
                st.write(f"**{mention}** → `{pred}`")
