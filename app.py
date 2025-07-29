import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("News Article Classifier:Fake or Real")
st.markdown("Enter a news article below to check if it's *Fake* or *Real*.")

text = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        vector = vectorizer.transform([text]).toarray()
        prediction = model.predict(vector)[0]
        result = "Real News" if prediction == 1 else "Fake News"
        st.success(f"Prediction: *{result}*")
