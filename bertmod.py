import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertEncoder, BertSelfAttention, BertSelfOutput, BertModel, BertConfig, BertPooler


##This implements a copy of the original attention mechnism to run simultaneously, then at the end, the outputs are joined together
class DualBertAttention(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        
        self.attention1 = BertSelfAttention(config)
        self.attention2 = BertSelfAttention(config)
        
        self.output1 = BertSelfOutput(config)
        self.output2 = BertSelfOutput(config)
        
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attn_output1 = self.attention1(hidden_states, attention_mask, head_mask)[0]  # Unpacking tuple
        attn_output1 = self.output1(attn_output1, hidden_states)

        attn_output2 = self.attention2(hidden_states, attention_mask, head_mask)[0]  # Unpacking tuple
        attn_output2 = self.output2(attn_output2, hidden_states)

        dual_attention_output = F.relu(attn_output1 + attn_output2)
        return dual_attention_output
    
##Implements the dual attention in the Neural Network
class DualBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DualBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


##Moves the data through the Neural Network
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,)


##Implments the outcome from the DualBertLayer to encode the data from the DualBertLayer Class
class DualBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([DualBertLayer(config) for _ in range(config.num_hidden_layers)])


##Implements the model with the DualBertEncoder
class DualBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = DualBertEncoder(config)
        self.pooler = BertPooler(config)  # Add this line

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        # Follow the same structure as the original BertModel
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  **kwargs)

        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output)


##Initialize a classifier between the 10 different options (3rd grade to 12th grade).

class BertClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim=64, output_dim=6):
        super().__init__()
        self.bert = bert_model
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :]  # outputs[0] = last_hidden_state
        return self.classifier(cls_embedding)

