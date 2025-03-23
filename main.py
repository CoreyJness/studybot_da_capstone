import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from predictor import predict_grade_level
from bertmod import DualBertModel, BertClassifier

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# Load model and tokenizer only once using Streamlit's caching
@st.cache_resource
def load_model_and_tokenizer():   
    # Load BERT config and initialize custom dual attention model
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert = DualBertModel(config)
    
    # Add classification head and load saved weights
    bert_classifier = BertClassifier(bert, 64, 10).to(device)
    bert_classifier.load_state_dict(torch.load("Bert_Classifier.pt", map_location=torch.device("cpu")))
    bert_classifier.eval()
    
    # Return the complete model and tokenizer
    return bert_classifier, tokenizer

# Cached model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit App UI
st.title("📚 StudyBot")
st.write("Are you smarter than a fifth grader? Enter a question and I’ll guess the grade level (3rd–12th).")

# Text input field for user question
question = st.text_input("💬 Enter your question here:")

# Button to trigger prediction
if st.button("🧠 Assess"):
    if question.strip() == "":
        # Handle blank input
        st.warning("Please enter a question first.")
    else:
        # Run prediction (predictor.py handles logic)
        grade = predict_grade_level(model, tokenizer, question, device)
        st.session_state["prediction"] = grade
        st.success(f"📊 Predicted Grade Level: {grade}")