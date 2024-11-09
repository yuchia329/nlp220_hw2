import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def downloadNLTK():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(articles)]

def preProcessText(text, isalpha=False, stopwords=False):
    # downloadNLTK()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()] if isalpha else tokens  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words] if stopwords else tokens  # Remove stopwords
    return " ".join(tokens)

def feature1(x):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    # vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=False, stopwords=False) for review in x]
    return x, vectorizer

def feature2(x):
    # vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=True, stopwords=True) for review in x]
    return x, vectorizer

def feature3(x):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), analyzer='word', ngram_range=(1, 1), max_features=99999)
    # vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer='word', ngram_range=(1, 1), max_features=99999)
    x = [preProcessText(review, isalpha=False, stopwords=False) for review in x]
    return x, vectorizer

def select_feature(x, feature_id):
    if feature_id == 1:
        x, vectorizer = feature1(x)
    elif feature_id == 2:
        x, vectorizer = feature2(x)
    else:
        x, vectorizer = feature3(x)
    return x, vectorizer