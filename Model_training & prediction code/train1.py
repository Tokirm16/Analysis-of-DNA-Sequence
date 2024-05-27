import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump

# Define the function to extract k-mers from a DNA sequence
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Load the dataset (assuming 'human_data.txt' is available in the specified path)
human_data = pd.read_table('human_data.txt')

# Process the dataset to extract k-mers
human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
human_data = human_data.drop('sequence', axis=1)
human_texts = [' '.join(words) for words in human_data['words']]

# Vectorize the k-mers using CountVectorizer
cv = CountVectorizer(ngram_range=(4, 4))
X = cv.fit_transform(human_texts)
y_data = human_data.iloc[:, 0].values

# Train the Naive Bayes classifier
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X, y_data)

# Save the trained classifier and CountVectorizer using joblib
dump(classifier, 'classifier.joblib')
dump(cv, 'count_vectorizer.joblib')
print('success')
