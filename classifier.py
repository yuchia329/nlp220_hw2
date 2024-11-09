from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

def selectClassifier(name={"svm", "decision_tree", "logistic_regression"}, best_hyper_param=False):
    if name == "svm":
        if best_hyper_param:
            classifier = LinearSVC(C=10, max_iter=1e-5, random_state=420)
        else:    
            classifier = LinearSVC(random_state=420)
    elif name == 'logistic_regression':
        classifier = LogisticRegression()
    elif name == "decision_tree":
        if best_hyper_param:
            classifier = DecisionTreeClassifier(max_features='sqrt', random_state=42)
        else:    
            classifier = DecisionTreeClassifier(random_state=42)
    else:
        classifier = OneVsRestClassifier(LinearSVC(random_state=420))
    return classifier

def searchGridFeature(modelName):
    model = selectClassifier(modelName)
    match modelName:
        case "svm":
            param_grid = { 
                'penalty': ['l1', 'l2'],
                'loss': ['hinge', 'squared_hinge'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [100000],
                'tol': [1e-4, 5e-4, 1e-5]
            }
        case "decision_tree":
            param_grid = { 'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ["sqrt", "log2"],
            }
        case "logistic_regression":
            param_grid = { 'penalty': ['l1', 'l2', 'elasticnet'],
                'tol': [1e-4, 5e-4, 1e-5, 5e-5, 1e-6],
                'C': [0.1, 1, 10, 100],
                'class_weight': ['class_weight', None],
                'solver': ["lbfgs", "liblinear", 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            }
        case _:
            param_grid = {}
    return model, param_grid