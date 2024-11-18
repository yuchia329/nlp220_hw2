import numpy as np
import torch
import random
import argparse
from load_data import processData, splitData
from feature import select_feature
from classifier import selectClassifier, searchGridFeature
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

def setSeed():
    seed = 490
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def searchGrid(model, X_train, X_vali, y_train, y_vali):
    model, param_grid = searchGridFeature(model)
    grid = GridSearchCV(model, param_grid, refit = True, verbose=0)
    # grid = HalvingGridSearchCV(model, param_grid, resource='n_estimators', max_resources=100, random_state=0)
    grid.fit(X_train, y_train)
    print(grid.best_params_) 
    grid_predictions = grid.predict(X_vali)
    p, r, f1, _ = precision_recall_fscore_support(y_true=y_vali, y_pred=grid_predictions, average='macro')
    print(f'macro precision: {p} recall: {r} f1: {f1}')

def main():
    setSeed()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default='data/ecommerceDataset.csv', nargs='?', help="Path to the training data CSV file")
    parser.add_argument("model", type=str, default='svm', nargs='?', help="choice of models: 'svm', 'logistic_regression', 'decision_tree'")
    parser.add_argument("feature", type=int, default=1, nargs='?', help="choice of features: 1, 2, 3")
    args = parser.parse_args()
    train_file = args.train_file
    model_name = args.model
    feature_id = args.feature
    print(train_file, model_name, feature_id)
    x, y, categoryDict = processData(train_file)
    x, vectorizer = select_feature(x, feature_id)
    X = vectorizer.fit_transform(x).toarray()
    x_train, x_vali, x_test, y_train, y_vali, y_test = splitData(X,y)    
    searchGrid(model_name, x_train, x_vali, y_train, y_vali)

if __name__ == "__main__":
    main()