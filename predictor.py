import torch
from transformers import AutoModel, AutoTokenizer
from bertmod import BertClassifier, DualBertModel, BertConfig
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = BertClassifier(DualBertModel(BertConfig()))


state_dict = torch.load("Bert_Classifier.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()


def predict_grade_level(model, tokenizer, question, device='cuda'):
    """
    Predicts and prints the grade level of a given question using the trained model.

    Args:
        model (torch.nn.Module): The trained classifier model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess input.
        question (str): The question string to classify.
        device (str): 'cuda' or 'cpu'

    Returns:
        str: Predicted grade level label (e.g., "5th Grade")
    """
    # Mapping index to grade label
    index_to_grade = {
        0: "3rd Grade", 1: "4th Grade", 2: "5th Grade", 3: "6th Grade", 4: "7th Grade",
        5: "8th Grade", 6: "9th Grade", 7: "10th Grade", 8: "11th Grade", 9: "12th Grade"
    }

    # Tokenize the input question
    encoded_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=32)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(encoded_input["input_ids"], encoded_input["attention_mask"])
        _, predicted_class = torch.max(logits, dim=1)

    grade = index_to_grade[predicted_class.item()]
    print(f"Predicted Grade Level: {grade}")
    return grade

