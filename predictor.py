import torch

def predict_grade_level(model, tokenizer, question, device='cpu'):
##Convert the output number to the correct grade
    index_to_grade = {
        0: "3rd Grade", 1: "4th Grade", 2: "5th Grade", 3: "6th Grade", 4: "7th Grade",
        5: "8th Grade", 6: "9th Grade", 7: "10th Grade", 8: "11th Grade", 9: "12th Grade"
    }

    ##Tokenize the question
    encoded_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=32)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    model.eval()
    ##Run the prediction
    with torch.no_grad():
        logits = model(encoded_input["input_ids"], encoded_input["attention_mask"])
        _, predicted_class = torch.max(logits, dim=1)
    
    ##Return the predicted class converted to the correct grade
    grade = index_to_grade[predicted_class.item()]
    print(f"Predicted Grade Level: {grade}")
    return grade