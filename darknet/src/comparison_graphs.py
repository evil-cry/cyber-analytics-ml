import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

class ComparisonGraphs:
    '''
        Contains the functions for the creation of the creation graphs
        for important metrics
    '''
    def __init__(self, model_results: dict):
        self.results = model_results

    def plot_roc_curves(self, model_name: str = None, class_label=None):
        '''
        Plot ROC for a single model and single class (if specified),
        or all models + micro-average if no args given.
        
        Params:
          model_name  – name of one model (must match a key in self.results). 
                        If None, loops over all models.
          class_label – the class value to plot (e.g. 0, 1, 2…). 
                        If None and multiclass, plots micro-average only.
        '''
        
        # This all assumes we are doing the OneVsRestClassifier for SVM
        plt.figure(figsize=(12, 6))
        
        if model_name in self.results:
            models = [model_name]
        else:
            self.results.keys()
        
        for m in models:
            result = self.results[m]
            y_true = np.array(result['y_true'])
            y_prob = np.array(result['y_prob'])
            classes = np.unique(y_true)

            # binary vs multiclass switch
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                # binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob.ravel())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.2f})")

            else:
                # binarize labels into one-hot
                y_true_bin = label_binarize(y_true, classes=classes)

                if class_label in classes:
                    # plot only the requested class
                    i = list(classes).index(class_label)
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(
                        fpr, tpr,
                        label=f"{m} – class {class_label} (AUC={roc_auc:.2f})"
                    )
                else:
                    # no specific class then plot micro-average
                    fpr, tpr, _ = roc_curve(
                        y_true_bin.ravel(),
                        y_prob.ravel()
                    )
                    roc_auc = auc(fpr, tpr)
                    plt.plot(
                        fpr, tpr,
                        linestyle='--',
                        label=f"{m} – micro-avg (AUC={roc_auc:.2f})"
                    )
                
        plt.plot([0, 1], [0, 1], linestyle='-', color='red', alpha=0.5)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_precision_recall_curves(self):
        '''
            Generate the precison and recall curves of the model
        '''
        pass

    def plot_runtime_comparison(self):
        '''
            Bar chart comparing the training times of each model
        '''
        pass

    def plot_metric_comparison(self, metric: str):
        '''
            Input the metric you would like to compare and create a 
            chart comparing the metric against all models
        '''
        pass
    