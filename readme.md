# Machine Learning
This repository is a collection of various machine learning projects.

## Spam Detector Classification
This project provides an easy way to implement and test spam detection using classifiers.
Current iteration realizes Naive Bayes and K-Nearest Neighbor classifiers.
The [report](spam_classification/docs/report.pdf) shows the full process of creating them, as well as various experimentation stages with classifier parameters.
Final results are shown using neat graphs.

### Setup
1. Install all the libraries specified in [requirements.txt](spam_classification/docs/requirements.txt)
2. If needed, adjust the _classifiers_ dictionary or the data path in _main()_ in [main.py](spam_classification/classifier/main.py). This is not needed for the assignment - everything is already set up by default.
3. The script will show accuracy, precision, recall and F1 scores for the given classifiers. It also puts the results [results.txt](spam_classification/docs/results.txt)
4. Note that you will need additional libraries not included in _requirements.txt_ if you plan to run value search or plotting methods. These are included in [optional_requirements.txt](classifier/optional_requirements.txt)

```sh
pip install -r spam_classification/docs/requirements.txt
python3 spam_classification/classifier/main.py
```

### Value Search and Graphs
1. If you need to find a parameter value that works best for a classifier, you are out of luck - the code was refactored and value tester no longer works. _make_graph()_ still works for old testing data.
2. Once testing is done, [make_graph.py](spam_classification/classifier/make_graph.py) provides a way to plot line graphs based on test results. Disclaimer: the method currently only looks at the first value that it finds to be changing in a subset of the test results. It first looks at Stop Word ‰ Removed. If it changes, it will not look at other parameters! If not, it goes one by one through parameters until it finds the first one that changes.  