import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

class ComparisonGraphs:
    '''
        Contains the functions for the creation of the creation graphs
        for important metrics
    '''
    def __init__(self, model_results: dict):
        self.results = model_results

    def calculate_metrics(self, y_true, y_pred, y_prob):
        '''
        Calculate metrics from raw predicitions
        '''
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        return metrics

    def plot_roc_curves(self, model_name: str = None, class_label=None, save_path: str = "darknet/graphs/roc_curve.png", figsize=(12, 6)):
        '''
        Plot ROC for a single model and single class (if specified),
        or all models + micro-average if no args given.
        
        Params:
          model_name  – name of one model (must match a key in self.results). 
                        If None, loops over all models.
          class_label – the class value to plot (e.g. 0, 1, 2…). 
                        If None and multiclass, plots micro-average only.
        '''
        print("Plotting ROC curves...")
        
        # This all assumes we are doing the OneVsRestClassifier for SVM
        plt.figure(figsize=figsize)
        
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
        plt.savefig(save_path)
        plt.show()
        
    def plot_precision_recall_curves(self, model_name: str = None, class_label=None, save_path: str = "darknet/graphs/precision_recall_curve.png", figsize=(12, 6)):
        '''
        Generate the precison and recall curves of the model
        '''

        print("Plotting Precision-Recall curves...")
        plt.figure(figsize=figsize)
    
        # Determine which models to plot
        if model_name is not None and model_name in self.results:
            models = [model_name]
        else:
            models = list(self.results.keys())
        
        for m in models:
            result = self.results[m]
            y_true = np.array(result['y_true'])
            y_prob = np.array(result['y_prob'])
            classes = np.unique(y_true)
            
            # Binary vs multiclass switch
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_prob.ravel())
                avg_precision = average_precision_score(y_true, y_prob.ravel())
                plt.plot(recall, precision, lw=2, label=f"{m} (AP={avg_precision:.2f})")
            
            else:
                # Multiclass classification - binarize labels into one-hot
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=classes)
                
                if class_label in classes:
                    # Plot only the requested class
                    i = list(classes).index(class_label)
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    plt.plot(
                        recall, precision, lw=2,
                        label=f"{m} – class {class_label} (AP={avg_precision:.2f})"
                    )
                else:
                    # Calculate micro-average precision-recall curve
                    precision = dict()
                    recall = dict()
                    average_precision = dict()
                    
                    for i in range(len(classes)):
                        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                        average_precision[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    
                    # Compute micro-average PR curve
                    precision["micro"], recall["micro"], _ = precision_recall_curve(
                        y_true_bin.ravel(), y_prob.ravel())
                    average_precision["micro"] = average_precision_score(
                        y_true_bin, y_prob, average="micro")
                    
                    plt.plot(
                        recall["micro"], precision["micro"], lw=2,
                        linestyle='--',
                        label=f"{m} – micro-avg (AP={average_precision['micro']:.2f})"
                    )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
    def plot_runtime_comparison(self, figsize=(12, 6), save_path="darknet/graphs/runtime_comparison.png"):  
        '''
        Bar chart comparing the training times of each model
        '''
        print("Plotting runtime comparison...")

        plt.figure(figsize=figsize)

        models = list(self.results.keys())
        times = [self.results[m]['train_time'] for m in models]

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        bars = plt.bar(models, times, color=colors[:len(models)], edgecolor='black', linewidth=1.2)

        # Add time values as text on top of bars
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + max(times)*0.02,
                    f'{time_val:.4f}s', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # Add grid lines for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust y-axis for text labels
        plt.ylim(0, max(times) * 1.15)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.xlabel("Models")
        plt.ylabel("Training Time (seconds)")
        plt.title("Model Training Time Comparison")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def plot_metric_comparison(self, metric_name: str = None, figsize=(12, 6), save_path: str = "darknet/graphs/metric_comparison.png"):
        '''
        Input the metric you would like to compare and create a 
        chart comparing the metric against all models
        '''
        plt.figure(figsize=figsize)
    
        models = []
        metric_v = []

        for model, metrics in self.results.items():
            if metric_name == 'train_time' and 'train_time' in self.results:
                models.append(model)
                metric_v.append(self.results['train_time'])

            elif 'y_true' in metrics and 'y_pred' in metrics:
                y_true = np.array(metrics['y_true'])
                y_pred = np.array(metrics['y_pred'])
                y_prob = np.array(metrics['y_prob'], None)

                metrics = self.calculate_metrics(y_true, y_pred, y_prob)

                if metric_name in metrics:
                    models.append(model)
                    metric_v.append(metrics[metric_name])

                else:
                    print(f"Metric '{metric_name}' not found for model '{model}'. Skipping...")

            else:
                print(f"Model '{model}' does not have the required metrics. Skipping...")

        df = pd.DataFrame({
            'Model': models,
            metric_name: metric_v
        })

        df = df.sort_values(by=metric_name, ascending=False)

        sns.set_style("whitegrid")

        ax = sns.barplot(
            x='Model',
            y=metric_name,
            data=df,
            palette='viridis'
        )

        plt.title(f"{metric_name.capitalize()} Comparison Across Models")
        plt.xlabel("Models")
        plt.ylabel(metric_name.capitalize())
        plt.xticks(rotation=45)
        
        for i, v in enumerate(metric_v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()