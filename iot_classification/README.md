## Decision Trees and Random Forests
This iteration of the codebase implements Internet of Thing device classification using Decision Trees and Random Forests.
The [report](iot_classification/docs/report.pdf) shows valuable insight on the final results and the process of creation.

### Setup
1. Install all the libraries specified in [requirements.txt](iot_classification/docs/requirements.txt)
2. If needed, unzip the iot_data corpus archive in 'iot_classification/corpus/' (press unzip here). The final path should be 'iot_classification/corpus/iot_data/iot_data/(folders)'
3. The script will create a Random Forest using the best hyperparameters and display the accuracy report .
4. To see all the graphs, go to 'iot_classification/graphs/'
4. To run the project ensure you are in the root directory (the directory containing the iot_classification folder) and run the following in the terminal:

```sh
pip install -r iot_classification/docs/requirements.txt
python3 iot_classification/src/classify.py
```

### Hyperparameter Search
To make graphs, first run the main classifier as instructed above. Then, change the save number in [make_graph.py](iot_classification/src/make_graph.py) main function and run it. 