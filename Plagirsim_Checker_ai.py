!pip install transformers datasets scikit-learn pandas numpy torch

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/kaggle

!unzip "/content/drive/MyDrive/kaggle/mit-plagiarism-detection-dataset.zip" -d /content/plagiarism_dataset

!head /content/plagiarism_dataset/train_snli.txt

import pandas as pd

data_path = "/content/plagiarism_dataset/train_snli.txt"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Sentence1", "Sentence2", "Label"])

df.head()

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df["Sentence1"] = df["Sentence1"].apply(clean_text)
df["Sentence2"] = df["Sentence2"].apply(clean_text)

df.head()

!pip install scikit-learn pandas numpy nltk

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")

from google.colab import drive
drive.mount('/content/drive')

data_path = "/content/drive/MyDrive/kaggle/train_snli.txt"
df = pd.read_csv(data_path, sep="\t")

if df.columns[0] != "Sentence1":
    df.columns = ["Sentence1", "Sentence2", "Label"]

tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

custom_stopwords = {"would", "could", "also", "one", "get", "us"}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    tokens = tokenizer.tokenize(text)  
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in custom_stopwords]  
    return " ".join(tokens)

df["Sentence1"] = df["Sentence1"].apply(clean_text)
df["Sentence2"] = df["Sentence2"].apply(clean_text)

df.dropna(subset=["Label"], inplace=True)

df["Combined"] = df["Sentence1"] + " " + df["Sentence2"]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strat_split.split(df["Combined"], df["Label"]):
    X_train, X_test = df["Combined"].iloc[train_index], df["Combined"].iloc[test_index]
    y_train, y_test = df["Label"].iloc[train_index], df["Label"].iloc[test_index]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = SGDClassifier(loss="hinge", penalty="l2", alpha=0.00005, max_iter=1500, tol=1e-4, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized SLM Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000]
}

grid_search = GridSearchCV(SGDClassifier(), param_grid, cv=5, verbose=1)
grid_search.fit(X_train_tfidf, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = SGDClassifier(alpha=0.0001, max_iter=2000)
best_model.fit(X_train_tfidf, y_train)

y_pred = best_model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))

data_path = '/content/drive/MyDrive/kaggle/mit-plagiarism-detection/train_snli.txt'

if os.path.exists(data_path):
    print(f"File found at: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    print(df.head())
else:
    print(f"File not found at: {data_path}. Please check the path and file name.")

joblib.dump(model, '/content/drive/MyDrive/kaggle/optimized_svm_model.pkl')

joblib.dump(vectorizer, '/content/drive/MyDrive/kaggle/tfidf_vectorizer.pkl')

model = joblib.load('/content/drive/MyDrive/kaggle/optimized_svm_model.pkl')
vectorizer = joblib.load('/content/drive/MyDrive/kaggle/tfidf_vectorizer.pkl')

new_data = [
    ["A person is riding a horse.", "A man is on a horse."],
    ["Children are playing in the park.", "There are kids playing soccer."]
]

df_new = pd.DataFrame(new_data, columns=["Sentence1", "Sentence2"])

df_new["Combined"] = df_new["Sentence1"] + " " + df_new["Sentence2"]

X_new_tfidf = vectorizer.transform(df_new["Combined"])

predictions = model.predict(X_new_tfidf)

for i, (sentence_pair, prediction) in enumerate(zip(new_data, predictions)):
    print(f"Sentence Pair {i+1}: {sentence_pair}")
    print(f"Prediction: {'Same' if prediction == 1 else 'Different'}\n")

print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred))

def predict_sentence_pair(sentence1, sentence2):
    combined = [sentence1 + " " + sentence2]
    transformed_input = vectorizer.transform(combined)

    prediction = model.predict(transformed_input)

    return "Same" if prediction == 1 else "Different"

print(predict_sentence_pair("The dog is playing in the yard.", "A dog is running around the yard."))
