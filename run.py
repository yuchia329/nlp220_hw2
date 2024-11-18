import numpy as np
import torch
import random
import argparse
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize, LabelBinarizer
from util import calculate_time
from plot import plotConfusionMatrix, plotPrecisionRecall, plotROC
from load_data import processData, splitData
from feature import select_feature
from classifier import selectClassifier


def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@calculate_time
def trainModel(model, x_train, y_train):
    model.fit(x_train, y_train)


@calculate_time
def predictModel(model, x_test):
    return model.predict(x_test)


def runModel(x, y, categoryDict, plot_matrix, name, feature_id, best_hyper=False):
    x, vectorizer = select_feature(x, feature_id)
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_vali, x_test, y_train, y_vali, y_test = splitData(X, y)
    classifier = selectClassifier(name, best_hyper)
    trainModel(classifier, x_train, y_train)
    y_pred = predictModel(classifier, x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)
    print(f'none p: {p}  r: {r} f1: {f1}')
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=y_test, y_pred=y_pred, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print('plot_matrix: ', plot_matrix)
    if plot_matrix:
        filename = name + '_feature' + str(feature_id)
        plotConfusionMatrix(y_test, y_pred, classifier, filename, categoryDict)


def runOneVsRest(x, y, categoryDict, plot_matrix, name, feature_id):
    n_classes = len(np.unique(y))
    x, vectorizer = select_feature(x, feature_id)
    X = vectorizer.fit_transform(x).toarray()
    y = label_binarize(y, classes=[0, 1, 2, 3])
    x_train, x_vali, x_test, y_train, y_vali, y_test = splitData(X, y)
    classifier = selectClassifier(name)
    trainModel(classifier, x_train, y_train)
    y_score = classifier.predict(x_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_pred = predictModel(classifier, x_test)
    # from sklearn.metrics import f1_score
    # f1_macro = f1_score(y_test, y_pred, average='macro')
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_score)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=y_test, y_pred=y_score, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')
    # print(f1_macro)
    # accuracy = accuracy_score(y_test, y_score)
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_score)
    # print(f'none p: {p}  r: {r} f1: {f1}')
    # p, r, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_score, average='macro')
    # print(f'macro precision: {p} recall: {r} f1: {f1}')
    # print("Classification Report:\n", classification_report(y_test, y_score))
    # print('plot_matrix: ', plot_matrix)
    # plotPrecisionRecall(classifier, categoryDict, y_test, y_score)
    # plotROC(n_classes, categoryDict, y_onehot_test, y_score)


def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default='data/ecommerceDataset.csv', nargs='?',
                        help="Path to the training data CSV file")
    parser.add_argument("plot_matrix", type=bool, default=False, nargs='?',
                        help="Whether to plot confusion matrix")
    args = parser.parse_args()
    train_file = args.train_file
    plot_matrix = args.plot_matrix
    x, y, categoryDict = processData(train_file)

    # runModel(x, y, categoryDict, plot_matrix, 'svm', 1)
    # runModel(x, y, categoryDict, plot_matrix, 'svm', 2)
    # runModel(x, y, categoryDict, plot_matrix, 'svm', 3)

    # runModel(x, y, categoryDict, plot_matrix, 'decision_tree', 1)
    # runModel(x, y, categoryDict, plot_matrix, 'decision_tree', 2)
    # runModel(x, y, categoryDict, plot_matrix, 'decision_tree', 3)

    # runModel(x, y, categoryDict, plot_matrix, 'logistic_regression', 1)
    # runModel(x, y, categoryDict, plot_matrix, 'logistic_regression', 2)
    # runModel(x, y, categoryDict, plot_matrix, 'logistic_regression', 3)

    runOneVsRest(x, y, categoryDict, plot_matrix, 'oneVsRest', 1)


if __name__ == "__main__":
    main()
