import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import os
import gdown

# Google Drive download setup
MODEL_FILE = "bert_classifier.pth"
GDRIVE_URL = "https://drive.google.com/uc?id=1DJGpKDKKGD4RTG0FkAvxMDH-xYb2jhSP"

# Download the model if not exists
if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model..."):
        gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)

# Define the BERTClassifier class
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Settings
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BERTClassifier(bert_model_name, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Define prediction function
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return "positive" if preds.item() == 1 else "negative"

# Streamlit interface
st.title("Team: AMMM - Deep Learning")

user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input, model, tokenizer, device)
        st.success(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter a valid review!")
