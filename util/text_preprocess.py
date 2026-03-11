import re
import nltk
import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))


def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove numbers and symbols
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub('[^a-z]', ' ', text)

    # tokenize
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # stemming
    words = [ps.stem(w) for w in words]

    return " ".join(words)