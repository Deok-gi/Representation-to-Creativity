from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
import torch
from transformers.modeling_longformer import LongformerSelfAttention
import pandas as pd
from sklearn.decomposition import PCA

class BertLongSelfAttention(LongformerSelfAttention):
    """
    Custom self-attention layer that uses Longformer's efficient attention mechanism.
    This allows BERT to handle longer sequences (up to 4096 tokens).
    """
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class BertLong(BertModel):
    """
    Extended BERT model with Longformer attention mechanism.
    Replaces standard BERT self-attention with efficient long-range attention.
    """
    def __init__(self, config):
        super().__init__(config)
        # Replace each layer's self-attention with Longformer attention
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model = BertLong.from_pretrained('../longformer-bert-base-4096')
model.to(device)  # Move model to device
model.eval()  # Set to evaluation mode (disables dropout, etc.)

tokenizer = BertTokenizerFast.from_pretrained('../longformer-bert-base-4096')

# Load essay data
essay_df = pd.read_csv('../datasets/D3.csv', encoding='utf-8')
docs = essay_df['essay'].tolist()

outputs = []

# Process each document
for doc in tqdm(docs, desc="Processing essays"):
    with torch.no_grad():  # Disable gradient computation for inference
        # Tokenize input text
        encoded_input = tokenizer(
            doc, 
            padding='max_length', 
            max_length=4096, 
            truncation=True,  # Truncate if text exceeds max_length
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        # Set global attention on [CLS] token (first token)
        # 2 = global attention in Longformer
        attention_mask[:, 0] = 2

        # Get model output (last hidden states)
        output = model(input_ids, attention_mask=attention_mask)[0]

        # Move output back to CPU and append
        outputs.append(output.squeeze(0).cpu())

# Stack all outputs into a single tensor
final_output = torch.stack(outputs, dim=0)

final_output  = final_output.view(final_output.shape[0], -1)

pca = PCA(n_components=2)

projected_data = pca.fit_transform(final_output.detach().numpy())

torch.save(projected_data, '../pt/D3_e2v_2d.pt')