import numpy as np
import pandas as pd
import pickle

# Define function to extract k-mers from DNA sequences
def extract_kmers(sequence, k=4):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers

# Define function to predict promoter
def predict_promoter(input_sequence, vocabulary, model, k=4):
    input_kmers = extract_kmers(input_sequence, k)
    input_vector = np.zeros(len(vocabulary))
    for kmer in input_kmers:
        if kmer in vocabulary:
            j = vocabulary.index(kmer)
            input_vector[j] += 1
    prediction = model.predict([input_vector])
    return prediction[0]

# Load the trained model and vocabulary
with open('model.pkl', 'rb') as f:
    nb = pickle.load(f)

with open('vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

# Prompt the user for input
user_input = input("Enter a DNA sequence: ")

# Predict promoter
result = predict_promoter(user_input, vocabulary, nb)

# Display prediction result
if result == '+':
    print(" sequence is predicted to be a promoter.")
else:
    print("sequence is predicted not to be a promoter.")
