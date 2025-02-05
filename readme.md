## Spam Detector Classifiers

This project implements spam detection using Naive Bayes and K-Nearest Neighbor classifiers.
The report shows the process of creating them, and various experimentation with classifier settings.
Final results are shown using neat  graphs.
The code is very granular and allows to easily implement and test additional classifiers.

## Setup
1. Install all the libraries specified in _classifier/requirements.txt_
2. If needed, adjust the _classifiers_ dictionary or the data path in _main()_ in _classifier/main.py_. This is not needed for the assignment - everything is already set up.
3. The script will show accuracy, precision, recall and F1 scores for the given classifiers. It also puts the results in _classifier/results.txt_

```sh
pip install -r classifier/requirements.txt
python3 classifier/main.py
```

## Value Search
1. If you need to find a parameter value that works best for a classifier, use _find_value()_
2. Comment out _run_tests()_ in _classifier/main.py_ and uncomment and existing _find_value()_ call or create a new one.
3. The method will iterate over the value ranges provided by you and print F1 scores. It also appends the results to _classifier/values.txt_