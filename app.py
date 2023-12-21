import numpy as np
from flask import Flask, request, render_template
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder="template", static_folder='staticFiles')

# Load models from files
model = load_model('build2.h5')

# Load the pre-trained tokenizer
with open('tokenizer.pkl', 'rb') as g:
    tokenizer = pickle.load(g)

# Assuming max_sequence_len is defined, adjust it based on your actual configuration
max_sequence_len =  124 

def Predict_Next_Words(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word

    return seed_text  # Add this line to return the result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    enter_text = request.form.get('text', '')
    next_words = int(request.form.get('next_words', 1))  # Assume a default of 1 next word

    if not enter_text:
        return render_template('index.html', Enter_text="No text provided", predicted_word="")

    # Use the provided logic for prediction
    result = Predict_Next_Words(enter_text, next_words)

    return render_template('index.html', Enter_text=enter_text, predicted_word=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)