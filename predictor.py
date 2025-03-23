import torch

def predict_grade_level(model, tokenizer, question, device='cpu'):
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
    index_to_grade = {
        0: "3rd Grade", 1: "4th Grade", 2: "5th Grade", 3: "6th Grade", 4: "7th Grade",
        5: "8th Grade", 6: "9th Grade", 7: "10th Grade", 8: "11th Grade", 9: "12th Grade"
    }

    encoded_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=32)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    model.eval()
    with torch.no_grad():
        logits = model(encoded_input["input_ids"], encoded_input["attention_mask"])
        _, predicted_class = torch.max(logits, dim=1)

    grade = index_to_grade[predicted_class.item()]
    print(f"Predicted Grade Level: {grade}")
    return grade



