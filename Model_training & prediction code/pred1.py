import numpy as np
import pandas as pd
from joblib import load

# Load the trained classifier and CountVectorizer
classifier = load('classifier.joblib')
cv = load('count_vectorizer.joblib')

# Define the function to extract k-mers from a DNA sequence
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Function to predict the class for a given DNA sequence
def predict_class(dna_sequence, fitted_cv, fitted_classifier, min_length=8):
    try:
        # Check if the input sequence meets the minimum length requirement
        if len(dna_sequence) < min_length:
            print("Error: DNA sequence is too short. Please provide a sequence with at least {} characters.".format(min_length))
            return None

        # Extract k-mers from the input sequence
        input_kmers = getKmers(dna_sequence)

        # Vectorize the input k-mers using the fitted CountVectorizer
        input_vectorized = fitted_cv.transform([' '.join(input_kmers)])

        # Predict the class using the fitted classifier
        predicted_class = fitted_classifier.predict(input_vectorized)

        return predicted_class[0]

    except ValueError as e:
        print("Error:", e)
        return None  # Or handle the error differently

# Input from the user
while True:
    user_input_sequence = input("Enter a DNA sequence: ")

    # Remove white spaces from the user input sequence
    user_input_sequence = ''.join(user_input_sequence.split())

    # Check if the input sequence contains valid DNA characters
    if all(base in 'ACGTatgc' for base in user_input_sequence):
        predicted_class = predict_class(user_input_sequence, cv, classifier)

        if predicted_class is not None:
            print("Predicted Class:", predicted_class)
            break
        else:
            print("Error in predicting class.")
    else:
        print("Invalid DNA sequence. Please enter a sequence containing only A, C, G, and T.")
