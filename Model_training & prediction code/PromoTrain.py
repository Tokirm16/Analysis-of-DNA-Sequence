import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
columns = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names=columns)

# Define function to extract k-mers from DNA sequences
def extract_kmers(sequence, k=4):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers

# Extract k-mers from DNA sequences
data['Kmers'] = data['Sequence'].apply(lambda x: extract_kmers(x, k))

# Create a vocabulary of unique k-mers
vocabulary = set()
for kmers in data['Kmers']:
    vocabulary.update(kmers)

# Convert vocabulary to a sorted list
vocabulary = sorted(list(vocabulary))

# Create feature matrix using k-mer counting
X = np.zeros((len(data), len(vocabulary)))
for i, kmers in enumerate(data['Kmers']):
    for kmer in kmers:
        j = vocabulary.index(kmer)
        X[i, j] += 1

# Define target variable
y = data['Class']
nb = GaussianNB()
nb.fit(X, y)

# Save the trained model and vocabulary using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(nb, f)

with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)
