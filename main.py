import streamlit as st
import torch
from transformers import AutoTokenizer, BertModel, BertConfig
from predictor import predict_grade_level
from bertmod import DualBertModel, BertClassifier
from transformers import BertModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load BERT config
config = BertConfig.from_pretrained("bert-base-uncased")

# Initialize DualBertModel and load BERT weights
dual_bert = DualBertModel(config)

# Load pretrained standard BERT to copy weights
pretrained_bert = BertModel.from_pretrained("bert-base-uncased")
dual_bert.embeddings = pretrained_bert.embeddings
dual_bert.encoder.load_state_dict(pretrained_bert.encoder.state_dict(), strict=False)
dual_bert.pooler = pretrained_bert.pooler

# Initialize classifier
model = BertClassifier(bert=dual_bert, hidden_dim=64, num_classes=10)


# Load trained weights
state_dict = torch.load("Bert_Classifier.pt", map_location=torch.device("cpu"))



model.eval()

# Load tokenizer and model once
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    classifier = model
    classifier.load_state_dict(torch.load("Bert_Classifier.pt", map_location=torch.device("cpu")))
    classifier.eval()
    return classifier, tokenizer

# Load once
model, tokenizer = load_model_and_tokenizer()

## Streamlit UI
st.title("📚 StudyBot: Grade-Level Question Classifier")
st.write("Are you smarter than a fifth grader? Enter a question and I’ll guess the grade level(3rd-12th).")

question = st.text_input("💬 Enter your question here:")

if st.button("🧠 Assess"):
    if question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        grade = predict_grade_level(model, tokenizer, question, device="cpu")
        st.success(f"📊 Predicted Grade Level: {grade}")
        st.session_state["prediction"] = grade