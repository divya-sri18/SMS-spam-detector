
import pickle

from util.text_preprocess import clean_text


model = pickle.load(open("models/spam_model.pkl","rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))


def predict_message(message):

    clean = clean_text(message)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0][1]

    label = "Spam" if prediction == 1 else "Not Spam"

    print("Original:", message)
    print("Cleaned:", clean)

    return label, round(probability*100,2)

