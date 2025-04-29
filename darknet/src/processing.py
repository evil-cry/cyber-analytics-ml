'''
AI Usage Statement
Tools Used: ChatGPT (o4-mini)
- Usage: Discussing ideas for encoding each column of the dataset.
- Verification: No code was given to me by the AI. All the ideas were bounced around and verified against my knowledge of the course material.

- Usage: Minor help with DataFrame reading and manipulation.
- Verification: pandas and glob documentation

Prohibited Use Compliance: Confirmed
'''

import pandas as pd
import numpy as np
import glob

from pandas import DataFrame

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

class Model:
    def __init__(self, model, name, data):
        try:
            self.name = name
            self.model = model
            self.X_train = data.X_train_scaled
            self.Y_train = data.Y_train
            self.X_test = data.X_test_scaled
            self.Y_test = data.Y_test

            self._train()
            self.predictions = self._classify()

        except (KeyError, AttributeError) as e:
            raise KeyError(f"Data does not contain a required key. Perpahs it's None? - {e}.")

    def _train(self):
        print(f'Training {self.name}...')
        self.model.fit(self.X_train, self.Y_train)

    def _classify(self):
        return self.model.predict(self.X_test)
    
    def evaluate(self, print_report = True):
        report = classification_report(self.Y_test, self.predictions)

        if print_report:
            print(f'Evaluating {self.name} - ')
            print()
            print(report)

            cm = sklearn_confusion_matrix(self.Y_test, self.predictions)
            print(f'Confusion matrix - \n{cm}')
            print()

        return report

class Data:
    def __init__(self, filepaths = 'darknet/corpus/parts/*.csv', kwargs = {}):
        self.glob: list = glob.glob(filepaths)

        if not self.glob:
            raise FileNotFoundError(f'No files found in {filepaths}')

        self.data: DataFrame = None
        self.kwargs = kwargs
        self.le = LabelEncoder()

        self._read()
        self._extract_features(self.kwargs)

    def _read(self):
        try:
            dataframes = [pd.read_csv(fp, delimiter=',') for fp in self.glob]
            self.data = pd.concat(dataframes, ignore_index=True)

            self.data.columns = self.data.columns.str.lower() # set columns to lowercase for easier access
            self.data = self.data.map(lambda x: str(x).lower() if isinstance(x, str) else x) # set all rows to lowercase if they are strings

            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%d/%m/%Y %I:%M:%S %p') 
            self.data['hour'] = self.data['timestamp'].dt.hour # these are the most useful ones - day of the week and hour of the day. We don't need much else.
            self.data['weekday'] = self.data['timestamp'].dt.weekday

            # one hot encode the protocols - make a separate column for each one
            self.data = pd.get_dummies(self.data, columns=['protocol'], prefix='protol')

            # string ip doesn't give much to the ai - even similar ips would be perceived as fully different enties. 
            # additionally, this split might give the ai more insight into the ip, such as seeing private and public ips.
            self.data['s1'], self.data['s2'], self.data['s3'], self.data['s4'] = self.split_ip(self.data['src ip'])
            self.data['d1'], self.data['d2'], self.data['d3'], self.data['d4'] = self.split_ip(self.data['dst ip'])
    
            self.drop_columns([
                'src ip', 'dst ip', 'timestamp',
                'flow id', # this column is simply a concat of the src and dst ip and ports
            ])

            self._cleanup()

        except FileNotFoundError as e:
            print(f'One of the files was not found - {e}')
            exit()
        
    def _cleanup(self):
        '''
        Cleans the features based on the following generic rules:
        - drop columns with only one unique value
        - replace infinities with None
        - drop Nones
        - convert data to numeric types if possible
        '''

        # drop columns with only one unique value
        self.data = self.data.loc[:, self.data.nunique() > 1]
        # replace infinities with None
        self.data.replace([np.inf, -np.inf], None, inplace=True)
        # drop Nones
        self.data.dropna(inplace=True)

        # explicitly convert data to correct types
        for column in self.data.columns:

            # exception-driven programming for the win
            try:
                self.data[column] = pd.to_numeric(self.data[column])
            except ValueError:
                pass  # if the column is not numeric, skip it

    def _extract_features(self, kwargs):
        '''
        Extract features from the dataset.
        '''

        if not isinstance(self.data, DataFrame):
            return None
        
        X = self.data.drop(columns=['label'], axis=1)
        if not kwargs.get('include_families', False):
            X.drop(columns=['family'], axis=1, inplace=True) 

        Y = self.data['label']
        Y = self.le.fit_transform(Y)

        maximum_features = kwargs.get('maximum_features', 5000)
        if len(X.columns) > maximum_features:
            # remove all columns after max
            X = X.iloc[:, :maximum_features]

        test_size = kwargs.get('test_size', 0.3)
        random_state = kwargs.get('random_state', 228)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        scaler = RobustScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
    def drop_columns(self, columns: list):
        '''
        Drop a column from the dataset
        '''
        if isinstance(self.data, DataFrame):
            self.data.drop(columns, axis=1, inplace=True)

    def print_unique_column_values(self, column: str):
        '''
        Print the unique values in a column
        '''
        if isinstance(self.data, DataFrame) and column in self.data.columns:
            print(f'Unique values in {column} - ')
            print(self.data[column].unique())

    def analyze_columns(self):
        '''
        For each column in the dataset, print:
        - data type
        - for numeric columns with ≤10 unique values, all unique values
        - for other numeric columns, print min and max values
        - for non-numeric columns, print the number of unique values
        and other useful information that I didn't document here
        '''
        dataframe = self.data

        if not isinstance(dataframe, DataFrame):
            return
        
        datatypes = set()
        counts = {}

        for column in dataframe.columns:
            dtype = dataframe[column].dtype
            datatypes.add(dtype)

            unique_count = len(dataframe[column].unique())
            counts[column] = unique_count
            
            print(f'{column} - {dtype} (dtype) - {unique_count} (unique count)')

            if np.issubdtype(dtype, np.number): # if it's a number
                if unique_count <= 10:
                    unique_values = sorted(dataframe[column].unique())
                    print(f'Unique values: {unique_values}')
                else:
                    min_val = dataframe[column].min()
                    max_val = dataframe[column].max()
                    print(f'Min/Max: {min_val} to {max_val}')
            
            else:
                if unique_count <= 10:
                    unique_values = sorted(dataframe[column].unique())
                    print(f'Unique values: {unique_values}')

            print()

            # graphs.plot_histogram('darknet/graphs/preprocessing', self.data, column)
            # graphs.plot_boxplot('darknet/graphs/preprocessing', self.data, column)

        print(f'Columns - {len(dataframe.columns)}')
        print(f'Unique counts - {counts}')
        print(f'Unique datatypes - {datatypes}')
    
    @staticmethod
    def split_ip(ip: pd.Series) -> list:
        '''
        Split a Series object ip into 4 integers and return them in a tuple
        '''
        ip1 = ip.str.split('.', expand=True)[0].astype(int)
        ip2 = ip.str.split('.', expand=True)[1].astype(int)
        ip3 = ip.str.split('.', expand=True)[2].astype(int)
        ip4 = ip.str.split('.', expand=True)[3].astype(int)
        
        return ip1, ip2, ip3, ip4