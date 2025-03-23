import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from predictor import predict_grade_level
from bertmod import DualBertModel, BertClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Load tokenizer and model once
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Rebuild full model
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert = DualBertModel(config)
    bert_classifier = BertClassifier(bert_model=bert, hidden_dim=64, output_dim=10)

    # Load the model state dict
    bert_classifier.load_state_dict(torch.load("bert_classifier.pt", map_location=device))

    # Move to device and set to eval mode
    bert_classifier.to(device)
    bert_classifier.eval()

    # Must match training architecture
    bert_classifier = BertClassifier(bert, 64, 10).to(device)
    bert_classifier.load_state_dict(torch.load("Bert_Classifier.pt", map_location=torch.device("cpu")))
    bert_classifier.eval()
    

    return bert_classifier, tokenizer


# Load once
model, tokenizer = load_model_and_tokenizer()

## Streamlit UI
st.title("📚 StudyBot")
st.write("Are you smarter than a fifth grader? Enter a question and I’ll guess the grade level (3rd–12th).")

question = st.text_input("💬 Enter your question here:")

if st.button("🧠 Assess"):
    if question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        grade = predict_grade_level(model, tokenizer, question, device)
        st.session_state["prediction"] = grade
        st.success(f"📊 Predicted Grade Level: {grade}")