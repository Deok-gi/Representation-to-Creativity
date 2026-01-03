import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from collections import Counter

# EM Clustering Section
# ====================================================================
def load_data(filepath):
    """Load essay data and scores from CSV file"""
    with open(filepath, 'r', encoding="UTF-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        docs, scores = [], []
        for line in reader:
            docs.append(line[1])
            scores.append(float(line[11]))
    return np.array(docs), scores

def perform_clustering(data, n_clusters=5):
    """Perform Gaussian Mixture clustering"""
    model = GaussianMixture(n_clusters, random_state=0)
    model.fit(data)
    labels = model.predict(data)
    return model, labels

def analyze_clusters(labels, scores, n_clusters):
    """Analyze cluster statistics"""
    cluster_scores = {k: 0.0 for k in range(n_clusters)}
    
    for label, score in zip(labels, scores):
        cluster_scores[label] += score
    
    unique_elements, counts = np.unique(labels, return_counts=True)
    counts_dict = dict(zip(unique_elements, counts))
    
    # Calculate average scores
    avg_scores = np.array([cluster_scores[i] / counts_dict[i] for i in range(n_clusters)])
    
    return counts_dict, avg_scores, cluster_scores

def visualize_clusters(data, labels, model, counts_dict):
    """Visualize clustering results"""
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title('EM Clustering Result')
    
    unique_labels, indices = np.unique(labels, return_index=True)
    for label, index in zip(unique_labels, indices):
        plt.scatter(data[index, 0], data[index, 1], c=label)
        plt.annotate(label, (data[index, 0], data[index, 1]), color='red', size=20)
    plt.show()

# Transformer Model Section
# ====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self._create_positional_encoding(position, d_model)

    def _get_angles(self, position, i, d_model):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def _create_positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(
            position=torch.arange(position, dtype=torch.float32).unsqueeze(1),
            i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
            d_model=d_model
        )

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        
        return angle_rads.unsqueeze(0).float()

    def forward(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.shape[0], :]

class Transformer(nn.Module):
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

    def prob_calc(self, encoder_x, decoder_x):
        encoder_x = self.embedding(encoder_x).squeeze(0)
        encoder_x = PositionalEncoding(encoder_x.shape[0], self.d_model)(encoder_x).squeeze(0)
        encoder_x = self.encoder(encoder_x)

        decoder_x = self.embedding(decoder_x).squeeze(0)
        decoder_x = PositionalEncoding(decoder_x.shape[0], self.d_model)(decoder_x).squeeze(0)
        decoder_y = self.decoder(decoder_x, encoder_x)

        return self.softmax(self.fc(decoder_y))

    def sequence_prob(self, sequence):
        inputs = sequence[sequence != 0]
        with torch.no_grad():
            output = self.prob_calc(inputs, inputs)
        return output[-1]

# Vocabulary and TF-IDF Section
# ====================================================================
def build_vocabulary(tokenized_data):
    """Build word-to-index and index-to-word mappings"""
    word2index = {'<PAD>': 0}
    index2word = {0: '<PAD>'}
    all_vocab = ['<PAD>']
    
    for sentence in tokenized_data:
        for word in sentence:
            if word not in word2index:
                index = len(word2index)
                word2index[word] = index
                index2word[index] = word
                all_vocab.append(word)
    
    return word2index, index2word, all_vocab

def get_top_tfidf_vocab(docs, all_vocab, word2index):
    """Get top vocabulary based on TF-IDF scores"""
    vectorizer = TfidfVectorizer(vocabulary=all_vocab)
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    tfidf_scores = zip(vectorizer.get_feature_names_out(), 
                       tfidf_matrix.sum(axis=0).tolist()[0])
    words = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    
    return [word2index[word[0]] for word in words]

# Creativity Analysis Section
# ====================================================================
def calculate_cluster_probs(model, sequences, top_vocab):
    """Calculate probability distribution for a cluster"""
    output = torch.zeros(model.embedding.num_embeddings)
    
    for sequence in sequences:
        output += model.sequence_prob(sequence)
    
    probs = (output / len(sequences)).tolist()
    return [probs[i] for i in top_vocab]

def calculate_cross_entropy(conventional_probs, candidate_probs, conventional_var, candidate_var):
    """Calculate cross-entropy between two probability distributions"""
    candidate_weight = candidate_var / (candidate_var + conventional_var)
    conventional_weight = conventional_var / (candidate_var + conventional_var)
    
    cross_entropy = 0
    for p1, p2 in zip(candidate_probs, conventional_probs):
        weighted_p1 = p1 * candidate_weight
        weighted_p2 = p2 * conventional_weight
        cross_entropy -= weighted_p1 * np.log(weighted_p2 + 1e-7)
    
    return cross_entropy

# Main Execution
# ====================================================================
def main():
    # Load data
    docs, scores = load_data('../datasets/D1.csv')
    print(f'Overall average: {sum(scores) / len(scores):.2f}')
    
    # Load pre-computed embeddings
    data = torch.load('../cache/D1_e2v_2d.pt')
    
    # Clustering
    K = 5
    model, labels = perform_clustering(data, K)
    print(f'Number of clusters: {model.n_components}')
    
    # Analyze clusters
    counts_dict, avg_scores, cluster_scores = analyze_clusters(labels, scores, K)
    
    # Print cluster information
    print(f'Total elements: {len(labels)}')
    for i in avg_scores.argsort():
        print(f'Cluster {i} - Count: {counts_dict[i]}, Avg creativity score: {avg_scores[i]:.2f}')
    
    # Calculate variances
    cluster_variances = [np.trace(cov) for cov in model.covariances_]
    print(f'\nVariances: {cluster_variances}\n')
    
    # Find most common (conventional) cluster
    most_common_cluster = Counter(labels).most_common(1)[0][0]
    print(f'Conventional cluster: Cluster {most_common_cluster}\n')
    
    # Visualize
    visualize_clusters(data, labels, model, counts_dict)
    
    # Load tokenized data
    with open('../cache/tokenized_data.pickle', 'rb') as f:
        tokenized_data = pickle.load(f)
    
    # Build vocabulary
    word2index, index2word, all_vocab = build_vocabulary(tokenized_data)
    print(f'Vocabulary size: {len(word2index)}')
    
    # Get top TF-IDF vocabulary
    top_vocab = get_top_tfidf_vocab(docs, all_vocab, word2index)
    
    # Initialize Transformer model
    TF_model = Transformer(vocab_size=len(word2index), d_model=512, nhead=8, num_layers=1)
    TF_model.load_state_dict(torch.load('../cache/transformer_weights.pth'))
    TF_model.eval()
    
    # Calculate conventional cluster probabilities
    conventional_idx = np.where(labels == most_common_cluster)[0]
    conventional_sequences = [torch.tensor([word2index[word] for word in tokenized_data[i]]) 
                       for i in conventional_idx]
    conventional_padded = pad_sequence(conventional_sequences, batch_first=True)
    conventional_probs = calculate_cluster_probs(TF_model, conventional_padded, top_vocab)
    
    # Calculate cross-entropy for each cluster
    cross_entropies = []
    for i in range(K):
        if i == most_common_cluster:
            cross_entropies.append(0)
            continue
        
        candidate_idx = np.where(labels == i)[0]
        candidate_sequences = [torch.tensor([word2index[word] for word in tokenized_data[j]]) 
                          for j in candidate_idx]
        candidate_padded = pad_sequence(candidate_sequences, batch_first=True)
        candidate_probs = calculate_cluster_probs(TF_model, candidate_padded, top_vocab)
        
        cross_entropy = calculate_cross_entropy(
            conventional_probs, candidate_probs, 
            cluster_variances[most_common_cluster], 
            cluster_variances[i]
        )
        
        print(f'Cross-Entropy between Conventional cluster {most_common_cluster} and Candidate cluster {i}: {cross_entropy:.4f}')
        cross_entropies.append(cross_entropy)
    
    # Find most creative cluster
    creative_cluster = cross_entropies.index(max(cross_entropies))
    print(f"\nCreative cluster: Cluster {creative_cluster}")

if __name__ == "__main__":
    main()