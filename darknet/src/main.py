import processing 
import comparison_graphs
import clustering # TODO - use clustering

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def compare_models(data):
    models = {
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
    }
    
    results = {}
    for name, model in models.items():
        clf = processing.Model(model, name, data)
        results[name] = clf.evaluate()
        
        results[name] = {
            'y_true': data.Y_test,
            'y_pred': clf.predictions,
            'y_prob': model.predict_proba(data.X_test_scaled),
            'train_time': clf.time
        }
        
    return results

def main():
    data = processing.Data('darknet/corpus/parts/*.csv')
    
    benign_kwargs = {'classify_families': True, 'label': 'benign'}
    benign = processing.Data('darknet/corpus/parts/*.csv', benign_kwargs)

    # TODO - make sure data analysis works
    # data.analyze_columns()
    
    
    # cluster = clustering.Cluster(benign, model_name='kmeans')
    # cluster.fit(n_clusters=3)
    # cluster.evaluate()
    # cluster.draw()
    
    #results = compare_models(data)
    #graphs  = comparison_graphs.ComparisonGraphs(results)
    #graphs.plot_runtime_comparison()
    
    results = compare_models(data)
    graphs = comparison_graphs.ComparisonGraphs(results)
    #graphs.plot_roc_curves(model_name="Random Forest", class_label=2)
    #graphs.plot_precision_recall_curves(model_name="Random Forest", class_label=2)
    #graphs.plot_runtime_comparison()
    graphs.plot_confusion_matrix()
    #graphs.plot_metric_comparison("accuracy")
    # graphs.plot_metric_comparison("accuracy")
    # graphs.plot_metric_comparison("precision")
    # graphs.plot_metric_comparison("recall")
    # graphs.plot_metric_comparison("f1")
    # graphs.plot_metric_comparison("fpr")

if __name__ == "__main__":
    main()