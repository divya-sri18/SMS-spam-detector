import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from util.text_preprocess import clean_text



df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({'ham':0,'spam':1})

df['clean'] = df['message'].apply(clean_text)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X = tfidf.fit_transform(df['clean']).toarray()

y = df['label']


X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

from sklearn.svm import LinearSVC

model = LinearSVC(class_weight="balanced")

model.fit(X_train,y_train)

from sklearn.metrics import classification_report

pred = model.predict(X_test)

print(classification_report(y_test,pred))


models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(class_weight="balanced")
}

for name,model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    print(name,"Accuracy:",acc)



pickle.dump(model,open("models/spam_model.pkl","wb"))
pickle.dump(tfidf,open("models/vectorizer.pkl","wb"))
