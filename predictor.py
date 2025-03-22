# %% [markdown]
# Now that we have the model trained, lets employ it. 

# %%
%run -i bertmod.py
import torch
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = BertClassifier(DualBertModel(BertConfig()))

state_dict = torch.load("Bert_Classifier.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)

model.to(device)

# %%
model.eval()

# %%
question = "Who is the main character in Oedipus"

# Tokenize and preprocess the input
encoded_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
encoded_input = encoded_input.to(device)  # Move to the correct device

# Make prediction
with torch.no_grad():
    logits = model(encoded_input["input_ids"], encoded_input["attention_mask"]) 

# Get predicted class
_, predicted_class = torch.max(logits, 1) 
print(f"Predicted Grade Level: {predicted_class.item()}")



