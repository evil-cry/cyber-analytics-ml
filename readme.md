## Spam Detector Classifiers

This project provides an easy way to implement and test spam detection using classifiers.
Current iteration realizes Naive Bayes and K-Nearest Neighbor classifiers.
The [report](report.pdf) shows the full process of creating them, as well as various experimentation stages with classifier parameters.
Final results are shown using neat graphs.

## Setup
1. Install all the libraries specified in [requirements.txt](classifier/requirements.txt)
2. If needed, adjust the _classifiers_ dictionary or the data path in _main()_ in [main.py](classifier/main.py). This is not needed for the assignment - everything is already set up.
3. The script will show accuracy, precision, recall and F1 scores for the given classifiers. It also puts the results [results.txt](classifier/results.txt)

```sh
pip install -r classifier/requirements.txt
python3 classifier/main.py
```

## Value Search
1. If you need to find a parameter value that works best for a classifier, use _find_value()_
2. Comment out _run_tests()_ in [main.py](classifier/main.py) and uncomment and existing _find_value()_ call or create a new one.
3. The method will iterate over the value ranges provided by you and print F1 scores. It also appends the results to [values.txt](classifier/values.txt)
4. Once testing is done, [make_graph.py](classifier/make_graph.py) provides a way to plot line graphs based on test results. Disclaimer: the method currently only looks at the first value that it finds to be changing in a subset of the test results. It first looks at Stop Word ‰ Removed. If it changes, it will not look at other parameters! If not, it goes one by one through parameters until it finds the first one that changes.  