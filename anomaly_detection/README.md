## Anomaly Detection using Clustering
This project provides an easy way to implement and test anomaly detection using clustering algorithms.
Current iteration implements DBScan and K-Means algorithms.
The [report](anomaly_detection/docs/report.pdf) shows the full process of creating them, as well as various experimentation stages with hyperparameters.
Final results are shown using somewhat neat graphs.

### Setup
1. Install all the libraries specified in [requirements.txt](anomaly_detection/docs/requirements.txt)
2. If needed, adjust the _data_ list in _main()_ in [main.py](anomaly_detection/clustering/main.py). This is not needed for the assignment - everything is already set up by default.
3. The script will show accuracy, true positive rate, false positive rate and F1 scores for the given clusterer. It also draws the graphs in [graphs](anomaly_detection/graphs)

```sh
pip install -r anomaly_detection/docs/requirements.txt
python3 anomaly_detection/clustering/main.py
```

### Value Search and Graphs
1. If you need to find a parameter value that works best for a clustering algorithm, utilize the other methods in main. 