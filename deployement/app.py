from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
from joblib import load


app = Flask(__name__)

# Load the trained classifier and CountVectorizer
modelsp = load('modelsp.joblib')
cvsp = load('vectorizersp.joblib')
classifierc = load('classifier.joblib')
cvc = load('count_vectorizer.joblib')

# information about dna classes
class_info = {
    0: {
        "class_name": "G protein-coupled receptors",
        "description": "G protein-coupled receptors (GPCRs) are integral membrane proteins that mediate cellular responses to diverse stimuli via G protein signaling. They are crucial drug targets, implicated in numerous diseases, and represent a major focus of drug discovery efforts for therapeutic intervention in conditions ranging from cardiovascular disorders to psychiatric illnesses.",
    },
    1: {
        "class_name": "Tyrosine kinases",
        "description": "Tyrosine kinases catalyze tyrosine phosphorylation, vital for cellular signaling and regulation of processes like growth and differentiation. In medicine, they are crucial targets for cancer therapy, with inhibitors designed to block their aberrant activity, highlighting their importance in disease treatment and drug development.",
    },
      2: {
        "class_name": "Tyrosine Phosphate",
        "description": "Tyrosine phosphorylation involves adding phosphate groups to tyrosine residues, crucial for regulating cellular processes. In medicine, it's implicated in diseases like cancer, driving research into tyrosine kinase inhibitors for targeted therapy, highlighting its importance in treatment strategies.",
    },
        3: {
        "class_name": "Synthetase",
        "description": "Synthetases are enzymes catalyzing molecule synthesis, vital for protein synthesis, DNA replication, and metabolism. In medicine, they're pivotal for drug development, gene therapy, and understanding disease mechanisms, serving as critical targets for research and therapeutic interventions.",
    },
          4: {
        "class_name": "Synthase",
        "description": "Synthases are enzymes driving molecule synthesis without ATP hydrolysis, pivotal in metabolic pathways like fatty acid and nucleotide biosynthesis. In medicine, they're essential for drug development targeting metabolic disorders, offering insights into disease mechanisms and potential therapies.",
    },
            5: {
        "class_nmae": "Ion channel",
        "description": "Ion channels are proteins in cell membranes controlling ion flow, crucial for nerve signaling, muscle function, and hormone release. In medicine, they're vital drug targets, with medications modulating their activity used to treat conditions like cardiac arrhythmias, epilepsy, and pain disorders, highlighting their importance in therapy and disease understanding.",
    },
              6: {
        "class_name": "Transcription factor",
        "description": "Transcription factors are proteins regulating gene expression by binding to DNA sequences, crucial for cellular processes like growth and differentiation. In medicine, their dysregulation contributes to diseases such as cancer and genetic disorders, driving research into targeted therapies and personalized medicine strategies.",
    }
}

# Species information dictionary
species_info = {
    "human": {  
        "image_url": "static/human.jpg",
        "description": "Humans are complex organisms belonging to the species Homo sapiens, characterized by bipedal locomotion and large brains. They possess intricate physiological systems, cognitive abilities, and social structures, contributing to their adaptability and dominance on Earth.",
        "habitat": "Humans are highly adaptable, inhabiting diverse environments from scorching deserts to frigid tundras, but primarily thrive in built structures and modified landscapes.",
        "more_info_url": "https://en.wikipedia.org/wiki/Human"
    },
    "Oak": { 
        "image_url": "static\oak.jpg",
        "description": "Oak trees, belonging to the genus Quercus, are hardwood trees with distinctive lobed leaves and produce acorns. They thrive in the northern hemisphere, providing habitat for wildlife and prized for their durable wood used in various industries.",
        "habitat": "Oaks are widespread trees found in temperate, subtropical, and Mediterranean regions, with specific species adapted to various climates and elevations.",
        "more_info_url": "https://en.wikipedia.org/wiki/Oak"
    },
     "chimpan": {  
        "image_url": "static\chimpanzee.jpg",
        "description": "Chimpanzees, our closest living relatives, are primates with expressive faces, opposable thumbs, and complex social structures. They inhabit forests in Africa, displaying tool use, communication through gestures and vocalizations, and sophisticated problem-solving skills.",
        "habitat": "Chimpanzees dwell in tropical rainforests and woodlands across central and western Africa, creating nests in tall trees for nightly rest.",
        "more_info_url": "https://en.wikipedia.org/wiki/Chimpanzee"
    },
      "dolphin": {  
        "image_url": "static\dolphin.jpg",
        "description": "Dolphins are marine mammals with streamlined bodies, dorsal fins, and blowholes for breathing air. Known for their intelligence, social behavior, and communication through clicks and whistles, they inhabit oceans worldwide.",
        "habitat": "Dolphins inhabit a wide range of marine environments, from tropical coasts and bays to cooler, temperate waters.  They can be found in both shallow and deep waters, depending on the species.",
        "more_info_url": "https://en.wikipedia.org/wiki/Dolphin"
    },
       "mushroom": {  
        "image_url": "static\mushrooms.jpg",
        "description": "Mushrooms are fungi, diverse in shape, size, and color, with spore-producing structures. They play vital ecological roles as decomposers and symbionts, while some species are edible and valued in cuisine.",
        "habitat": "Mushrooms aren't like animals; they decompose organic matter in damp, shady areas or partner with plants in forests.",
        "more_info_url": "https://en.wikipedia.org/wiki/Mushroom"
    },
}


def extract_kmers(sequence, k=4):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers

# Define the function to extract k-mers from a DNA sequence
def getKmers_classi(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Define the function to extract k-mers from a DNA sequence
def getKmers(sequence, size=8):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


# Define the function to find the species using a DNA sequence
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
    
    
    # this function ia used to predict the gene class of dna_sequence
def predict_dna_class(dna_sequence, fitted_cv, fitted_classifier):
    # Extract k-mers from the input sequence
    input_kmers = getKmers_classi(dna_sequence)

    # Vectorize the input k-mers using the fitted CountVectorizer
    input_vectorized = fitted_cv.transform([' '.join(input_kmers)])

    # Predict the class using the fitted classifier
    predicted_class = fitted_classifier.predict(input_vectorized)
    print("Predict class", predicted_class)

    return predicted_class[0]

# # Function to predict the class and probability scores for a given DNA sequence
# def predict_class_with_probabilities(dna_sequence, fitted_cv, fitted_classifier, min_length=8):
#     try:
        
#         # Extract k-mers from the input sequence
#         input_kmers = getKmers(dna_sequence)

#         # Vectorize the input k-mers using the fitted CountVectorizer
#         input_vectorized = fitted_cv.transform([' '.join(input_kmers)])

#         # Predict the class using the fitted classifier
#         predicted_class = fitted_classifier.predict(input_vectorized)

#         # Get the probability scores for each class
#         probabilities = fitted_classifier.predict_proba(input_vectorized)

#         return predicted_class[0], probabilities

#     except ValueError as e:
#         return "Error: {}".format(e), None  # Or handle the error differently
      

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/identify')
def identify():
    return render_template('identify.html')

@app.route('/promoter-check')
def promoterIndex():
    return render_template('promo.html')

@app.route('/about')
def about():
    return render_template('about.html')


# This route is used for Species prediction.
@app.route('/identi', methods=['POST'])
def identi():
    user_input_sequence = request.form.get('dna-sequence')
    user_input_sequence = ''.join(user_input_sequence.split())

    try:
        predicted_class = predict_class(user_input_sequence, cvsp, modelsp)

        # Retrieve species information from dictionary (if class exists)
        species_details = species_info.get(predicted_class)
        if species_details:
            image_url = species_details['image_url']
            description = species_details['description']
            habitat = species_details['habitat']
            more_info_url = species_details['more_info_url']
        else:
            # Handle case where predicted class is not found in the dictionary
            error_message = f"Species information not found for: {predicted_class}"
            return render_template('identify_result.html', predict_class=predicted_class, error=error_message)

    except Exception as e:  # Catch potential errors during prediction or info retrieval
        error_message = str(e)
        return render_template('identify_result.html', predict_class=None, error=error_message)

    return render_template('identify_result.html', predict_class=predicted_class,
                          image_url=image_url, description=description,
                          habitat=habitat,more_info_url=more_info_url)
   
   
# This route is responsible for promoter identification
@app.route('/promoter', methods=['POST'])
def promoter():
    
     # Load the trained model and vocabulary
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f) 
        
     # Preprocess the input DNA sequence
    seq = request.form.get('dna-sequence')
    input_kmers = extract_kmers(seq)
    input_vector = np.zeros(len(vocabulary))
    for kmer in input_kmers:
        if kmer in vocabulary:
            j = vocabulary.index(kmer)
            input_vector[j] += 1

    # Make predictions using the trained model
    prediction = model.predict([input_vector])
    result = "Promoter" if prediction == '+' else "Non-Promoter"

    # Render the result on the promoter_result.html page
    return render_template('promoter_result.html', result=result)
   
    


# This route is used for DNA classification into classes.
@app.route('/classi', methods=['POST'])
def classi():
    if request.method == 'POST':
        sequence = request.form['sequence']
        # print(sequence)
        predicted_class = predict_dna_class(sequence, cvc, classifierc)
                              
        class_details = class_info.get(predicted_class)
                   
        return render_template('classify_result.html', prediction=predicted_class, class_details=class_details)
        

if __name__ == '__main__':
    app.run(debug=True)
