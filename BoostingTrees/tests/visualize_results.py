import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_arr, tpr_arr = np.array(fpr_list), np.array(tpr_list)
    sorted_indices = np.argsort(fpr_arr)
    fpr_arr, tpr_arr = fpr_arr[sorted_indices], tpr_arr[sorted_indices]

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_arr, tpr_arr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
