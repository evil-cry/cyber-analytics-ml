import processing 

from sklearn.ensemble import RandomForestClassifier

def main():
    data_kwargs = {}
    data = processing.Data('darknet/corpus/darknet.csv', data_kwargs)

    forest_kwargs = {} 
    model = processing.Model(RandomForestClassifier(**forest_kwargs), data)
    model.evaluate()

if __name__ == "__main__":
    main()