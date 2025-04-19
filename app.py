import streamlit as st
import pickle

# Load model
model = pickle.load(open('sentiment_analysis.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

# Custom header
st.title('Sentiment Analysis Model')
st.markdown("""
**Team:** AMMM 
**Course:** Deep Learning
""")

# User input
review = st.text_input('Enter your review:')

# Prediction button
submit = st.button('Predict')

if submit:
    prediction = model.predict([review])

    if prediction[0] == 'positive':
        st.success('Positive Review')
    else:
        st.warning('Negative Review')
