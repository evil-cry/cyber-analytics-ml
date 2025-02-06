import re
import matplotlib.pyplot as plt
import numpy as np
import main as classifiers

def parse_data(line: str) -> tuple:
    '''
    Parses a line of text and extracts method, stop word top mille, parameters, and F1 score.
    A data point must be in the format: method(stop_word_top_mille, {params}): f1%.
    Here, method is the method name, both parenthesis and the percentage sign are literal characters. stop_word_top_mille is an integer, params is a dictionary, and f1 is a float.
    Params is a dictionary of str:int, where str is the parameter name and int is the parameter value. 
    @params:
        line (str): data point
    @returns:
        tuple: A tuple containing:
            method (str): The name of the method.
            stop_word_top_mille (int): The number of stop words removed per mille.
            params (dict): A dictionary of parameters.
            f1 (float): The F1 score.
    @exceptions:
        ValueError: If the line does not match the expected format.
    '''
    pattern = r'(\w+)\((\d+),\s*(\d+),\s*({.*})\):\s*([\d.]+)%'# This was made using GPT-4o
    match = re.match(pattern, line.strip())
    if not match:
        raise ValueError(f"Data format incorrect: {line}")
    
    method = match.group(1)
    stop_word_top_mille = int(match.group(2))
    min_count = int(match.group(3))
    params_str = match.group(4)
    f1 = float(match.group(5))

    params = eval(params_str)
    return method, stop_word_top_mille, min_count, params, f1

def determine_changing_key(data_points: list) -> str:
    '''
    Determines which key in the data points is changing.
    Returns the first changing parameter it finds.
    @params:
        data_points (list): A list of dictionaries, with the following key:value pairs:
            'method': str - The method name.
            'stop_word_top_mille': int - The number of stop words removed per mille.
            'params': dict - A dictionary of parameter names and values.
            'f1': float - The F1 score.
    @returns:
        str: The key that has changing values, or None if no keys have changing values.
    '''
    if not data_points:
        return None
    
    # If stop_words are changing, return that
    stop_values = [dp['stop_word_top_mille'] for dp in data_points]
    if len(set(stop_values)) > 1:
        return 'stop_word_top_mille'
    
    # Else, return the parameter
    param_keys = data_points[0]['params'].keys()
    for key in param_keys:
        values = [dp['params'].get(key, None) for dp in data_points]
        if len(set(values)) > 1:
            return key
    
    return None # No key is changing


def plot_tests() -> None:
    '''
    Reads test data from a file, parses it and generates graphs.
    Data points must be located in 'classifier/values.txt'.
    Distinct tests must be separated with two or more newlines.
    A data point must be in the format: method(stop_word_top_mille, {params}): f1%.
    Here, method is the method name, both parenthesis and the percentage sign are literal characters. stop_word_top_mille is an integer, params is a dictionary, and f1 is a float.
    Params is a dictionary of str:int, where str is the parameter name and int is the parameter value. 
    '''
    filename = 'classifier/values.txt'
    with open(filename, 'r') as file:
        content = file.read()
    
    tests = re.split(r'\n{2,}', content.strip()) # Test data is separated by two or more newlines
    
    for test_n, test in enumerate(tests, 1):
        lines = test.strip().split('\n')
        data_points = []
        
        for line in lines:
            try:
                method, stop_word_top_mille, min_count, params, f1 = parse_data(line)
                data_points.append({
                    'method': method,
                    'stop_word_top_mille': stop_word_top_mille,
                    'min_count': min_count,
                    'params': params,
                    'f1': f1
                })
            except ValueError as e:
                print(e)
                continue

        # Each test has only one changing key - get it and use it
        changing_key = determine_changing_key(data_points)
    
        if changing_key == 'stop_word_top_mille':
            x_label = 'Stop Word ‰ Removed'
            x_values = [dp['stop_word_top_mille'] for dp in data_points]
        else:
            x_label = changing_key
            x_values = [dp['params'][changing_key] for dp in data_points]
        
        y_values = [dp['f1'] for dp in data_points]
        
        sorted_pairs = sorted(zip(x_values, y_values))
        x_sorted, y_sorted = zip(*sorted_pairs)
        
        graph_label = input(f"Enter label for test {test_n} graph: ")
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted, y_sorted, marker='o', label=graph_label)

        # Dynamically adjust the axes scales
        # Use 10% margin on y-axis
        y_margin = (max(y_sorted) - min(y_sorted)) * 0.1

        # X axis shows all integer x values
        plt.xlim(min(x_sorted), max(x_sorted) + 1)
        plt.xticks(x_sorted)

        # Set some limits on the Y axis
        plt.ylim(min(y_sorted) - y_margin, max(y_sorted) + y_margin)


        plt.xlabel(x_label)
        plt.ylabel('F1 score (%)')
        plt.title(f'F1 score vs {x_label}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'graphs/{test_n}.png')
        plt.show()

def plot_learning_curve(method: callable, data: list, params: dict) -> None:
    '''
    Plots learning curves to detect overfitting/underfitting.
    @params:
        method (callable): The method to test.
        data (list): The data to test on.
        params (dict): Parameters for the method.
    '''

    # Turns out converting results into strings directly inside test methods isn't the best idea
    def percentage_to_float(str):
        return float(str.strip('%')) / 100.0
    
    # Create training sizes from 10% to 100% of data
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    
    full_train_size = int(len(data) * 0.8)
    full_test_data = data[full_train_size:] 
    
    for train_size in train_sizes:
        # Take a subset of training data
        subset_size = int(full_train_size * train_size)
        train_subset = data[:subset_size]
        
        # Get scores for training data
        tp, tn, fp, fn = method(train_subset, train_subset, **params)
        _, _, _, f1_train = classifiers.calculate_statistics(tp, tn, fp, fn)
        train_scores.append(percentage_to_float(f1_train))
        
        # Get scores for test data
        tp, tn, fp, fn = method(train_subset, full_test_data, **params)
        _, _, _, f1_test = classifiers.calculate_statistics(tp, tn, fp, fn)
        test_scores.append(percentage_to_float(f1_test))
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes * 100, train_scores, 'o-', label='Training score')
    plt.plot(train_sizes * 100, test_scores, 'o-', label='Test score')
    
    plt.xlabel('Training set size (%)')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curve - {method.__name__}')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(f'graphs/learning_curve_{method.__name__}.png')
    plt.show()

if __name__ == "__main__":
    data = "corpus/SMSSpamCollection"

    #plot_tests()

    '''
    I tried moving these two calls inside plot_learning_curve
    When I did that, the resulting plot was hugely different from the current one
    I could not figure out why that is happening - in both cases, the data is processed in the exact same way
    I am losing my mind over this
    '''

    '''
    stop_words = classifiers.find_stop_words(data, 5)
    train_data, test_data = classifiers.get_data(data, stop_words)
    plot_learning_curve(classifiers.test_nb, test_data + train_data, {})
    '''
    
    stop_words = classifiers.find_stop_words(data, 1, 0)
    train_data, test_data = classifiers.get_data(data, stop_words)
    plot_learning_curve(classifiers.test_nb, test_data + train_data, {})