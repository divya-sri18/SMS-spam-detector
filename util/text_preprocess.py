import re
import nltk
import nltk
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))


def clean_text(text):

    
    text = text.lower()

    
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub('[^a-z]', ' ', text)

    
    words = text.split()

    
    words = [w for w in words if w not in stop_words]

    
    words = [ps.stem(w) for w in words]

    return " ".join(words)
