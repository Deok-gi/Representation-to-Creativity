import torch
import torch.nn as nn
import torch.optim as optim
from konlpy.tag import Okt
import csv
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for Transformer models.
    Adds positional information to input embeddings using sine and cosine functions.
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """Calculate angles for positional encoding"""
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        Generate positional encoding matrix.
        Uses sine for even indices and cosine for odd indices.
        """
        angle_rads = self.get_angles(
            position=torch.arange(position, dtype=torch.float32).unsqueeze(1),
            i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
            d_model=d_model
        )

        # Apply sine function to even indices (2i)
        sines = torch.sin(angle_rads[:, 0::2])
        # Apply cosine function to odd indices (2i+1)
        cosines = torch.cos(angle_rads[:, 1::2])

        angle_rads = torch.zeros_like(angle_rads)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = angle_rads.unsqueeze(0)

        return pos_encoding.float().to(device)

    def forward(self, inputs):
        """Add positional encoding to input embeddings"""
        return inputs + self.pos_encoding[:, :inputs.shape[0], :]


def tokenize(text):
    """
    Tokenize Korean text using KoNLPy's Okt morphological analyzer.
    """
    okt = Okt()
    text = okt.normalize(text)
    tokens = okt.morphs(text)
    return tokens


def preprocess(text):
    """Preprocess text by tokenizing"""
    tokens = tokenize(text)
    return tokens


class Transformer(nn.Module):
    """
    Transformer model with encoder-decoder architecture.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_x, decoder_x):
        """
        Forward pass through the Transformer.
        Args:
            encoder_x: Input sequence for encoder
            decoder_x: Input sequence for decoder
        Returns:
            Output logits
        """
        # Encode input sequence
        encoder_x = self.embedding(encoder_x)
        encoder_x = encoder_x.squeeze(0)
        encoder_x = PositionalEncoding(encoder_x.shape[0], self.d_model)(encoder_x)
        encoder_x = encoder_x.squeeze(0)
        encoder_x = self.encoder(encoder_x)
        
        # Decode target sequence
        decoder_x = self.embedding(decoder_x)
        decoder_x = decoder_x.squeeze(0)
        decoder_x = PositionalEncoding(decoder_x.shape[0], self.d_model)(decoder_x)
        decoder_x = decoder_x.squeeze(0)
        decoder_y = self.decoder(decoder_x, encoder_x)

        # Project to vocabulary size
        y = self.fc(decoder_y)
        return y


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data from CSV file
print("Loading data...")
f = open('../datasets/D1.csv', 'r', encoding="UTF-8")
reader = csv.reader(f)
cnt = 0
docs = []
for line in reader:
    if cnt != 0:
        docs.append(line[1])
    else:
        cnt += 1
f.close()

# Preprocess and tokenize data
print("Preprocessing data...")
tokenized_data = [preprocess(sentence) for sentence in tqdm(docs)]

with open('../cache/tokenized_data.pickle', 'wb') as file:
    pickle.dump(tokenized_data, file)
    
# Build vocabulary
print("Building vocabulary...")
word2index = {}
index2word = {}

word2index['<PAD>'] = 0
index2word[0] = '<PAD>'

all_vocab = []

for sentence in tokenized_data:
    for word in sentence:
        if word not in word2index:
            all_vocab.append(word)
            index = len(word2index)
            word2index[word] = index
            index2word[index] = word

vocab_size = len(word2index)
print(f'Vocabulary size: {vocab_size}')

# Convert tokens to indices and pad sequences
sequences = [torch.tensor([word2index[word] for word in sentence]) for sentence in tokenized_data]
padded_sequences = pad_sequence(sequences, batch_first=True)

# Prepare training data (encoder input, decoder input, and target)
encoder_x_train = padded_sequences
decoder_x_train = padded_sequences[:, :-1]  # All tokens except last
target_train = padded_sequences[:, 1:]  # All tokens except first (shifted)

# Move data to device
encoder_x_train = encoder_x_train.to(device)
decoder_x_train = decoder_x_train.to(device)
target_train = target_train.to(device)

# Initialize model
print("Initializing model...")
d_model = 512  # Embedding dimension
nhead = 8  # Number of attention heads
num_layers = 1  # Number of encoder/decoder layers

model = Transformer(vocab_size, d_model, nhead, num_layers)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
print("Starting training...")
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    # Iterate through all training samples
    for i in tqdm(range(len(encoder_x_train)), desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()  # Reset gradients
        
        # Get single sample
        encoder_x_batch = encoder_x_train[i]
        decoder_x_batch = decoder_x_train[i]
        target_batch = target_train[i]
        
        # Forward pass
        output = model(encoder_x_batch, decoder_x_batch)
        loss = criterion(output, target_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Print epoch loss
    avg_loss = epoch_loss / len(encoder_x_train)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model weights
print("Saving model weights...")
torch.save(model.state_dict(), '../cache/transformer_weights.pth')
print("Training completed!")