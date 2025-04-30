import processing 
import comparison_graphs
import clustering # TODO - use clustering

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def compare_models(data):
    models = {
        'Random Forest': RandomForestClassifier(),
        #'SVM': OneVsRestClassifier(SVC(probability=True)), # Changed to OneVsRestClassifier for now
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
    data_kwargs = {}
    data = processing.Data('darknet/corpus/parts/*.csv', data_kwargs)

    # TODO - make sure data analysis works
    # data.analyze_columns()
    
    # cluster = clustering.Cluster(data, model_name='kmeans')
    # cluster.fit(n_clusters=4)
    # cluster.evaluate()
    # cluster.draw()

    results = compare_models(data)
    graphs = comparison_graphs.ComparisonGraphs(results)
    graphs.plot_roc_curves(model_name="Random Forest", class_label=2)
    graphs.plot_precision_recall_curves(model_name="Random Forest", class_label=2)
    graphs.plot_runtime_comparison()
    graphs.plot_metric_comparison("accuracy")

if __name__ == "__main__":
    main()