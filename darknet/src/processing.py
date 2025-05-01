'''
AI Usage Statement
Tools Used: ChatGPT (o4-mini)
- Usage: Discussing ideas for encoding each column of the dataset.
- Verification: No code was given to me by the AI. All the ideas were bounced around and verified against my knowledge of the course material.

- Usage: Help with DataFrame reading and manipulation.
- Verification: pandas and glob documentation

Prohibited Use Compliance: Confirmed
'''

import glob
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

class Model:
    def __init__(self, model, name, data, kwargs):
        try:
            self.name = name
            self.model = model
            self.kwargs = kwargs

            self.X_train, self.X_test, self.Y_train, self.Y_test = data.set_get_X_Y(
                kwargs.get('what_to_classify', 'class'),
                kwargs.get('scaler', StandardScaler()),
                kwargs.get('max_samples', -1)
            )

            start = time.time()
            self._train()
            end = time.time()
            self.time = end - start

            self._classify()

        except (KeyError, AttributeError) as e:
            raise KeyError(f"Data does not contain a required key. Perhaphs it's None? - {e}.")

    def _train(self):
        print(f'Training {self.name}...')
        self.model.fit(self.X_train, self.Y_train)

    def _classify(self):
        if hasattr(self.model, "predict_proba"):
            self.y_prob = self.model.predict_proba(self.X_test)
        else:
            self.y_prob = None

        self.predictions = self.model.predict(self.X_test)
    
    def evaluate(self, print_report = True):
        self.report = classification_report(self.Y_test, self.predictions)
        self.confusion_matrix = sklearn_confusion_matrix(self.Y_test, self.predictions)

        if print_report:
            print(f'It took {self.time:.2f} seconds to train {self.name}.\n')

            print(f'{self.name} Performance Metrics - ')
            print(f'\n{self.report}\n')

            print(f'{self.name} Confusion Matrix - ')
            print(f'\n{self.confusion_matrix}\n')
    
    def draw_confusion_matrix(self):
        save_path = f'darknet/graphs/{self.name}'
        os.makedirs(save_path, exist_ok=True)
        save_path = save_path + '/confusion_matrix.png'

        plt.figure(figsize=(8, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d')
        plt.title(f'{self.name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_path)

class Data:
    def __init__(self, filepaths = 'darknet/corpus/parts/*.csv', kwargs = {}):
        self.glob: list = glob.glob(filepaths)

        if not self.glob:
            raise FileNotFoundError(f'No files found in {filepaths}')

        self.data: DataFrame = None
        self.kwargs = kwargs

        self.le = LabelEncoder()

        self._read()

    def _read(self):
        try:
            dataframes = [pd.read_csv(fp, delimiter=',') for fp in self.glob]
            self.data = pd.concat(dataframes, ignore_index=True)

            # non-tor and non-vpn traffic are the same
            # use capital L because it isn't lowercased yet
            self.data = self.data[self.data['Label'] != 'non-tor']
            self.data.drop_duplicates(inplace=True) # drop duplicates just in case

            self.data.columns = self.data.columns.str.lower() # set columns to lowercase for easier access
            self.data = self.data.map(lambda x: str(x).lower() if isinstance(x, str) else x) # set all rows to lowercase if they are strings

            # change non-vpn and non-tor to benign
            # yes, non-tor doesn't exist anymore. yes, I wrote this comment instead of changing that.
            self.data['label'] = self.data['label'].apply(lambda x: x if x in ['vpn', 'tor'] else 'benign')

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

            self.benign = self.data[self.data['label'] == 'benign'].copy(deep=True)
            self.vpn = self.data[self.data['label'] == 'vpn'].copy(deep=True)
            self.tor = self.data[self.data['label'] == 'tor'].copy(deep=True)

            self.benign.drop(columns=['label'], axis=1, inplace=True)
            self.vpn.drop(columns=['label'], axis=1, inplace=True)
            self.tor.drop(columns=['label'], axis=1, inplace=True)

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

    def _extract_features(self, what_to_classify: str = 'class', scaler = StandardScaler(), max_samples = -1):
        '''
        Extract features from the dataset.
        '''

        if not isinstance(self.data, DataFrame):
            return None
        
        if max_samples != -1 and what_to_classify == 'class':
            # get the labels
            self.data = self.data.groupby('label').apply(
                # sample up to max_samples
                lambda x: x.sample(min(len(x), max_samples), random_state=228)
            # remove grouping
            ).reset_index(drop=True)
        
        if what_to_classify == 'class':
            X = self.data.drop(columns=['label'], axis=1)
            X.drop(columns=['family'], axis=1, inplace=True) 
            Y = self.data['label']

        elif what_to_classify == 'benign':
            X = self.benign
            X.drop(columns=['family'], axis=1, inplace=True)
            Y = self.benign['family']

        elif what_to_classify == 'vpn':
            X = self.vpn
            X.drop(columns=['family'], axis=1, inplace=True)
            Y = self.vpn['family']

        elif what_to_classify == 'tor':
            X = self.tor
            X.drop(columns=['family'], axis=1, inplace=True)
            Y = self.tor['family']

        Y = self.le.fit_transform(Y)

        test_size = self.kwargs.get('test_size', 0.3)
        random_state = self.kwargs.get('random_state', 228)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    def set_get_X_Y(self, what_to_classify: str = 'class', scaler = StandardScaler(), max_samples = -1):
        '''
        get the X and Y lists for training
        what_to_classify - if class, classify labels, if benign; vpn; or tor, classify families of said class
        scaler - what scaler to use. None for no scaling
        max_samples - maximum number of samples to use for training. 0 for all
        '''
        self._extract_features(what_to_classify, scaler, max_samples)

        if scaler:
            try:
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)
            except AttributeError as e:
                raise AttributeError(f'Scaler must be a sklearn scaler class - {e}')

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

        return X_train_scaled, X_test_scaled, self.Y_train, self.Y_test
        
        
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
        """
        Print a presentable summary of each column:
          - Data type and unique count
          - Numeric stats (mean, median, std, min, max, percentiles)
          - Top categories with percentages for non-numeric
        """
        df = self.data
        n_rows = len(df)
        benign = len(df[df['label'] == 'benign'])
        vpn = len(df[df['label'] == 'vpn'])
        tor = len(df[df['label'] == 'tor'])

        print('=== Dataset Summary ===')
        print(f'{n_rows} samples: {benign} benign, {vpn} vpn, {tor} tor. {df.shape[1]} columns.\n')
        print()

        for col in df.columns:
            series = df[col]
            dtype = series.dtype
            nunique = series.nunique(dropna=False)
            print(f"Column: {col!r} ({dtype}) — {nunique} unique")

            if np.issubdtype(dtype, np.number):
                # Numeric summary
                mean = series.mean()
                median = series.median()
                std = series.std()
                minimum = series.min()
                maximum = series.max()
                q25, q50, q75 = series.quantile([0.25, 0.5, 0.75])
                print(f"  • mean={mean:.3f}, median={median:.3f}, std={std:.3f}")
                print(f"  • range=[{minimum:.3f} → {maximum:.3f}]")
                print(f"  • 25%={q25:.3f}, 50%={q50:.3f}, 75%={q75:.3f}")

            else:
                # Top 5 values
                counts = series.value_counts(dropna=False)
                top8 = counts.iloc[:8]
                for val, cnt in top8.items():
                    pct = cnt / n_rows * 100
                    val_label = '<NaN>' if pd.isna(val) else val
                    print(f"  • {val_label!r}: {cnt} ({pct:.2f}%)")
                if nunique > 8:
                    print(f"  • ... and {nunique - 8} more unique values")

            print() 

    def analyze_columns_debug(self):
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