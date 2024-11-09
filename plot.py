from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np

def plotPrecisionRecall(classifier, categoryDict, y_test, y_score):
    precision = dict()
    recall = dict()
    n_classes = classifier.classes_
    for i in range(len(n_classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(categoryDict[i]))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig('OvR_PrecisionRecall.png')
    plt.clf()
    
def plotROCOld(classifier, categoryDict, y_test, y_score):
    fpr = dict()
    tpr = dict()
    n_classes = classifier.classes_
    for i in range(len(n_classes)):
        fpr[i], tpr[i], _ = roc_auc_score(y_test[:, i], y_score[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(categoryDict[i]))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.savefig('OvR_ROC.png')
    plt.clf()
    
def plotROC(n_classes, categoryDict, y_onehot_test, y_score):

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    from itertools import cycle
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = cycle(["tomato", "aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {categoryDict[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 3),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )
    plt.savefig('OvR_ROC.png')
    plt.clf()
    
def plotConfusionMatrix(y_true, y_pred, classifier, filename, categoryDict):
    if getattr(classifier, 'classes_', None) is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = classifier.classes_
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    labels = [categoryDict[index] for index in labels]
    print("labels: ", labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot().figure_.savefig(filename)