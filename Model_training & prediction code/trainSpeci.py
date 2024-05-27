import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from itertools import permutations
from tqdm import tqdm
from joblib import dump

# Function to extract k-mers from a DNA sequence
def getKmers(sequence, size=8):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Define function to parse genome file and create DataFrame
def parse_from_genome_file(filename, target_name, n=None):
    i = 0
    seqs = []
    target = []

    for seq_record in SeqIO.parse(filename, "fasta"):
        seqs.append(''.join(seq_record.seq))
        target.append(target_name)

        if n is not None:
            if i < n:
                i += 1
            if i >= n:
                break

    df = pd.concat([pd.DataFrame(seqs), pd.DataFrame(target)], axis=1)
    df.columns = ['seq', 'target']
    return df

# Read the genome files and create DataFrames
human_df = parse_from_genome_file("data/human_gene.fna", 'human', n=1000)
chimp_df = parse_from_genome_file('data/chimpan_gene.fna', 'chimpan', n=1000)
dolphin_df = parse_from_genome_file('data/dolphin_gene.fna', 'dolphin', n=1000)
oak_df = parse_from_genome_file('data/GCF_001633185.2_ValleyOak3.2_cds_from_genomic.fna', 'Oak', n=1000)
mushroom_df = parse_from_genome_file('data/mushroom_gene.fna', 'mushroom', n=1000)

# Concatenate all DataFrames into one
species_compare_df = pd.concat([human_df, chimp_df, dolphin_df, oak_df, mushroom_df], axis=0)

# Load the trained classifier and CountVectorizer
classifier = MultinomialNB(alpha=0.4)
cv = CountVectorizer(ngram_range=(4,4))

# Process the dataset to extract k-mers
species_compare_df['words'] = species_compare_df.apply(lambda x: getKmers(x['seq']), axis=1)
species_compare_df = species_compare_df.drop('seq', axis=1)
species_texts = [' '.join(words) for words in species_compare_df['words']]

# Vectorize the k-mers using the pre-trained CountVectorizer
X = cv.fit_transform(species_texts)
y_data = species_compare_df.iloc[:, 0].values

# Train the Naive Bayes classifier
classifier.fit(X, y_data)

# Save the trained model and CountVectorizer
dump(classifier, 'modelsp.joblib')
dump(cv, 'vectorizersp.joblib')
