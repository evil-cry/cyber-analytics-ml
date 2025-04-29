import processing 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def compare_models(data):
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
    }
    
    results = {}
    for name, model in models.items():
        clf = processing.Model(model, name, data)
        results[name] = clf.evaluate()

def main():
    data_kwargs = {}
    data = processing.Data('darknet/corpus/parts/*.csv', data_kwargs)

    # uncomment to analyze columns
    #data.analyze_columns()

    compare_models(data)
    

if __name__ == "__main__":
    main()