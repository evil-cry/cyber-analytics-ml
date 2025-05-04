import processing 
import comparison_graphs
import clustering # TODO - use clustering

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def compare_models(data):
    models = {
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'OvO Random Forest': OneVsOneClassifier(RandomForestClassifier(), n_jobs=-1),
        'OvA Random Forest': OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1),
    }

    model_kwargs = {
        'Random Forest': {'scaler': RobustScaler()},
        'KNN': {'scaler': RobustScaler()}, # it was found that additional PCA only hurts the model
        'SVM': {'scaler': StandardScaler(), 'pca':30}, # doesn't imrpove after 30. Using standardScaler() because robust completely freezes it
        'OvO Random Forest': {},
        'OvA Random Forest': {},
    }
    
    results = {}
    for name, model in models.items():
        current_kwargs = model_kwargs.get(name, {})
        clf = processing.Model(model, name, data, current_kwargs)
        results[name] = clf.evaluate()
        
        results[name] = {
            'y_true': data.Y_test,
            'y_pred': clf.predictions,
            'y_prob': clf.y_prob,
            'train_time': clf.time
        }
        
    return results

def main():
    data = processing.Data('darknet/corpus/parts/*.csv')
    
    benign_kwargs = {'classify_families': True, 'label': 'benign'}
    benign = processing.Data('darknet/corpus/parts/*.csv', benign_kwargs)

    #data.analyze_columns()
    
    '''
    cluster = clustering.Cluster(data, model_name='kmeans', kwargs={'what_to_classify': 'class'})
    cluster.fit(n_clusters=3)
    cluster.evaluate()
    cluster.draw(darkmode=False)

    cluster = clustering.Cluster(benign, model_name='kmeans', kwargs={'what_to_classify': 'benign'})
    cluster.fit(n_clusters=8)
    cluster.evaluate()
    cluster.draw(darkmode=False)
    '''
    
    results = compare_models(data)
    graphs = comparison_graphs.ComparisonGraphs(results)

    graphs.plot_roc_curves(model_name="Random Forest", darkmode=False)
    graphs.plot_precision_recall_curves(model_name="Random Forest", darkmode=False)
    graphs.plot_runtime_comparison(darkmode=False)

    graphs.plot_confusion_matrix(decoder=data.decode_labels)

    graphs.plot_metric_comparison("accuracy", darkmode=False)
    graphs.plot_metric_comparison("accuracy", darkmode=False)
    graphs.plot_metric_comparison("precision", darkmode=False)
    graphs.plot_metric_comparison("recall", darkmode=False)
    graphs.plot_metric_comparison("f1", darkmode=False)
    graphs.plot_metric_comparison("fpr", darkmode=False)

if __name__ == "__main__":
    main()