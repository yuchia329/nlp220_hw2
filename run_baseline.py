import polars as pl
import numpy as np
import torch
import random
import argparse
import nltk
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hinge_loss, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from util import calculate_time
from sklearn.multiclass import OneVsRestClassifier
import multiprocessing

def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def processData(file_path):
    df = pl.read_csv(file_path)
    df.columns = ["category", "text"]
    category = df.get_column('category').unique(maintain_order=True)
    df = df.drop_nulls()
    df = df.with_columns(pl.col("category").cast(pl.Categorical))
    df = df.with_columns(pl.col("category").to_physical().alias("category"))
    categoryIndex = df.get_column('category').unique(maintain_order=True)
    x = df['text'].to_numpy()
    y = df['category'].to_numpy()
    return x, y, dict(map(lambda i,j : (i,j) , categoryIndex,category))

def getLabels(file_path):
    df = pl.read_csv(file_path)
    df.columns = ["category", "text"]
    category = df.get_column('category').unique(maintain_order=True)
    df = df.with_columns(pl.col("category").cast(pl.Categorical))
    df = df.with_columns(pl.col("category").to_physical().alias("category"))
    categoryIndex = df.get_column('category').unique(maintain_order=True)
    return dict(map(lambda i,j : (i,j) , categoryIndex,category))
    

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

def plotMatrix(y_true, y_pred, classifier, filename, categoryDict):
    # print(getattr(classifier, 'classes_', None))
    if getattr(classifier, 'classes_', None) is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = classifier.classes_
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    labels = [categoryDict[index] for index in labels]
    print("labels: ", labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot().figure_.savefig(filename)

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

def selectClassifier(name={"svm", "decision_tree", "logistic_regression"}):
    if name == "svm":
        classifier = make_pipeline( LinearSVC(random_state=420))
    elif name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=42)
    elif name == "decision_tree":
        classifier = LogisticRegression()
    else:
        classifier = OneVsRestClassifier(LinearSVC(loss='squared_hinge', C=10, max_iter=100000, penalty='l2', random_state=420, tol=1e-4))
    return classifier


@calculate_time
def trainModel(model, x_train, y_train):
    model.fit(x_train, y_train)

@calculate_time
def predictModel(model, x_test):
    return model.predict(x_test)

def searchGridFeature(modelName):
    match modelName:
        case "svm":
            model = LinearSVC()
            param_grid = { 
                'penalty': ['l1', 'l2'],
                'loss': ['hinge', 'squared_hinge'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [100000],
                'tol': [1e-4, 5e-4, 1e-5]#, 5e-5, 1e-6],
            }
        case "decision_tree":
            model = DecisionTreeClassifier()
            param_grid = { 'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ["sqrt", "log2"],
            }
        case _:
            model = LogisticRegression()
            param_grid = { 'penalty': ['l1', 'l2', 'elasticnet'],
                'tol': [1e-4, 5e-4, 1e-5, 5e-5, 1e-6],
                'C': [0.1, 1, 10, 100],
                'class_weight': ['class_weight', None],
                'solver': ["lbfgs", "liblinear", 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            }
    return model, param_grid

@calculate_time
def searchGrid(model, X_train, X_test, y_train, y_test):
    model, param_grid = searchGridFeature(model)
    n_jobs = multiprocessing.cpu_count()-30
    print(n_jobs)
    grid = GridSearchCV(model, param_grid, refit = True, verbose=0, n_jobs=n_jobs)
    # grid = HalvingGridSearchCV(model, param_grid, resource='n_estimators', max_resources=100, random_state=0)
    grid.fit(X_train, y_train)
    print(grid.best_params_) 
    grid_predictions = grid.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=grid_predictions, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print(classification_report(y_test, grid_predictions))

@calculate_time
def runModel(x, y, categoryDict, plot_confusion, name, feature):
    
    if feature == 1:
        x, vectorizer = feature1(x)
    elif feature == 2:
        x, vectorizer = feature2(x)
    else:
        x, vectorizer = feature3(x)
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    x_test, x_vali, y_test, y_vali = train_test_split(
        x_test, y_test, test_size=0.33, random_state=42)
    classifier = selectClassifier(name)
    trainModel(classifier, x_train, y_train)
    print('validation')
    y_pred = predictModel(classifier, x_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_vali, y_pred=y_pred)
    print(f'none p: {p}  r: {r} f1: {f1}')
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_vali, y_pred=y_pred, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print("Classification Report:\n", classification_report(y_vali, y_pred))
    
    print('test')
    y_pred = predictModel(classifier, x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)
    print(f'none p: {p}  r: {r} f1: {f1}')
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if plot_confusion:
        filename = 'baseline_' + name + '_feature' + str(feature)
        plotMatrix(y_test, y_pred, classifier, filename, categoryDict)

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default='data/ecommerceDataset.csv', nargs='?',
                        help="Path to the training data CSV file")
    parser.add_argument("plot_confusion", type=bool, default=False, nargs='?',
                        help="Whether to plot confusion matrix")
    args = parser.parse_args()
    train_file = args.train_file
    plot_confusion = args.plot_confusion
    x, y, categoryDict = processData(train_file)
    runModel(x, y, categoryDict, plot_confusion, 'svm', 1)
    runModel(x, y, categoryDict, plot_confusion, 'svm', 2)
    runModel(x, y, categoryDict, plot_confusion, 'svm', 3)

    runModel(x, y, categoryDict, plot_confusion, 'decision_tree', 1)
    runModel(x, y, categoryDict, plot_confusion, 'decision_tree', 2)
    runModel(x, y, categoryDict, plot_confusion, 'decision_tree', 3)

    runModel(x, y, categoryDict, plot_confusion, 'logistic_regression', 1)
    runModel(x, y, categoryDict, plot_confusion, 'logistic_regression', 2)
    runModel(x, y, categoryDict, plot_confusion, 'logistic_regression', 3)
    
    # runModel(train_file, plot_confusion, 'svmOneVsRest', 1)
    
    

if __name__ == "__main__":
    main()