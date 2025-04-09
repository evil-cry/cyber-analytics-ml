'''
AI Usage Statement
Tools Used: Copilot (Claude 3.7 Sonnet)
- Usage: Making the graphs from the save file I made. Graph types were selected by me.
- Verification: The graphs look correct and match the data. The code was reviewed and rewritten to fit the needs.
Prohibited Use Compliance: Confirmed
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pickle
from classify import Save

def make_tree_graph_errors(pickle_path='iot_classification/docs/tuning/1.pklrick', output_dir='iot_classification/graphs'):
    '''
    Create error bar plots showing the mean accuracy with min and max ranges for each hyperparameter.
    '''

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f'Loading tree hyperparameter data from {pickle_path}...')
    with open(pickle_path, 'rb') as f:
        save_obj = pickle.load(f)
    
    # Convert tree data to DataFrame
    tree_data = save_obj.trees
    df = pd.DataFrame(tree_data, columns=['max_depth', 'min_node', 'feature_count', 'unused', 'accuracy'])
    # Drop the column that is used for processing in the main classify script
    df = df.drop('unused', axis=1)
    
    df = df.fillna(110)
    
    df_depth = df.copy()
    #df_depth = df_depth[df_depth['max_depth'] > 10]
    
    # Generate error bar plot for max_depth vs accuracy
    print('Creating max_depth error bar plot...')
    depth_stats = df_depth.groupby('max_depth')['accuracy'].agg(['mean', 'min', 'max']).reset_index()
    
    plt.figure(figsize=(12, 6))
    # Calculate asymmetric error bars (distance from mean to min/max)
    yerr = np.array([
        depth_stats['mean'] - depth_stats['min'],  # lower error
        depth_stats['max'] - depth_stats['mean']   # upper error
    ])

    average_of_average_std = depth_stats['mean'].std()
    print(average_of_average_std)
    
    plt.errorbar(
        depth_stats['max_depth'], 
        depth_stats['mean'], 
        yerr=yerr, 
        fmt='o',
        capsize=5, 
        elinewidth=1, 
        markersize=4, 
        color='blue'
    )
    
    # Show all x-tick values
    plt.xticks(depth_stats['max_depth'])
    
    plt.xlabel('max depth')
    plt.ylabel('Accuracy (mean with min/max range)')
    plt.title('Decision Tree: Maximum Depth vs Accuracy with Min/Max Range')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/max_depth_error_bars.png', dpi=300)
    plt.close()
    
    # Generate error bar plot for min_node vs accuracy
    print('Creating min_node error bar plot...')
    node_stats = df.groupby('min_node')['accuracy'].agg(['mean', 'min', 'max']).reset_index()
    node_stats = node_stats.sort_values('min_node')
    
    plt.figure(figsize=(12, 6))
    # Calculate asymmetric error bars
    yerr = np.array([
        node_stats['mean'] - node_stats['min'],  # lower error
        node_stats['max'] - node_stats['mean']   # upper error
    ])

    average_of_average_std = node_stats['mean'].std()
    print(average_of_average_std)
    
    plt.errorbar(
        node_stats['min_node'], 
        node_stats['mean'], 
        yerr=yerr, 
        fmt='o',
        capsize=5, 
        elinewidth=1, 
        markersize=4, 
        color='green'
    )
    
    # Show all x-tick values
    plt.xticks(node_stats['min_node'])
    
    plt.xlabel('min node')
    plt.ylabel('Accuracy (mean with min/max range)')
    plt.title('Decision Tree: Minimum Nodes vs Accuracy with Min/Max Range')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/min_node_error_bars.png', dpi=300)
    plt.close()
    
    # Generate error bar plot for feature_count vs accuracy
    print('Creating feature_count error bar plot...')
    feature_stats = df.groupby('feature_count')['accuracy'].agg(['mean', 'min', 'max']).reset_index()
    feature_stats = feature_stats.sort_values('feature_count')
    
    plt.figure(figsize=(12, 6))
    # Calculate asymmetric error bars
    yerr = np.array([
        feature_stats['mean'] - feature_stats['min'],  # lower error
        feature_stats['max'] - feature_stats['mean']   # upper error
    ])

    average_of_average_std = feature_stats['mean'].std()
    print(average_of_average_std)
    
    plt.errorbar(
        feature_stats['feature_count'], 
        feature_stats['mean'], 
        yerr=yerr, 
        fmt='o',
        capsize=5, 
        elinewidth=1, 
        markersize=4, 
        color='purple'
    )
    
    # Show all x-tick values
    plt.xticks(feature_stats['feature_count'])
    
    plt.xlabel('feature count')
    plt.ylabel('Accuracy (mean with min/max range)')
    plt.title('Decision Tree: Feature Count vs Accuracy with Min/Max Range')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_count_error_bars.png', dpi=300)
    plt.close()
    
    print(f'Min/max range visualizations saved to {output_dir}')

def make_tree_graph_other(pickle_path='iot_classification/docs/tuning/1.pklrick', output_dir='iot_classification/graphs'):
    '''
    Create individual line graphs showing the relationship between each tree hyperparameter and accuracy.
    '''
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f'Loading tree hyperparameter data from {pickle_path}...')
    with open(pickle_path, 'rb') as f:
        save_obj = pickle.load(f)
    
    # Convert tree data to DataFrame
    tree_data = save_obj.trees
    df = pd.DataFrame(tree_data, columns=['max_depth', 'min_node', 'feature_count', 'unused', 'accuracy'])
    # Drop the column that is used for processing in the main classify script
    df = df.drop('unused', axis=1)
    
    # Replace None values with 110 for the graph to work
    df = df.fillna(110)
    df = df[df['feature_count'] <= 20]
    
    # Generate line graph for max_depth vs accuracy
    print('Creating max_depth vs accuracy line graph...')
    depth_acc_df = df.groupby('max_depth', as_index=False)['accuracy'].mean()
    depth_acc_df = depth_acc_df.sort_values('max_depth')
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_acc_df['max_depth'], depth_acc_df['accuracy'], 'o-', linewidth=2, markersize=8, color='blue')

    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Maximum Depth vs Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/max_depth_vs_accuracy.png', dpi=300)
    plt.close()
    
    # Generate line graph for min_node vs accuracy
    print('Creating min_node vs accuracy line graph...')
    node_acc_df = df.groupby('min_node', as_index=False)['accuracy'].mean()
    node_acc_df = node_acc_df.sort_values('min_node')
    
    plt.figure(figsize=(12, 6))
    plt.plot(node_acc_df['min_node'], node_acc_df['accuracy'], 'o-', linewidth=2, markersize=8, color='green')
    
    plt.xlabel('Minimum Samples per Node')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Minimum Samples per Node vs Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/min_node_vs_accuracy.png', dpi=300)
    plt.close()
    
    # Generate line graph for feature_count vs accuracy
    print('Creating feature_count vs accuracy line graph...')
    feature_acc_df = df.groupby('feature_count', as_index=False)['accuracy'].mean()
    feature_acc_df = feature_acc_df.sort_values('feature_count')
    
    plt.figure(figsize=(12, 6))
    plt.plot(feature_acc_df['feature_count'], feature_acc_df['accuracy'], 'o-', linewidth=2, markersize=8, color='purple')
    
    plt.xlabel('Max Features')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Max Features vs Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_count_vs_accuracy.png', dpi=300)
    plt.close()
    
    print(f'Individual hyperparameter visualizations saved to {output_dir}')

def make_forest_graph(pickle_path='iot_classification/docs/tuning/1.pklrick', output_dir='iot_classification/graphs'):## 
    '''
    Create a line graph showing the relationship between number of trees and accuracy.
    '''
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f'Loading forest hyperparameter data from {pickle_path}...')
    with open(pickle_path, 'rb') as f:
        save_obj = pickle.load(f)
    
    # Convert forest data to DataFrame
    forest_data = save_obj.forests
    tree_acc_df = pd.DataFrame(forest_data, columns=['n_trees', 'accuracy'])
    
    # Handle potential duplicate entries by taking the mean accuracy for each tree count
    tree_acc_df = tree_acc_df.groupby('n_trees', as_index=False)['accuracy'].mean()
    
    # Sort by tree count
    tree_acc_df = tree_acc_df.sort_values('n_trees')
    
    # Set the minimum value for y-axis slightly below the minimum accuracy
    min_acc = tree_acc_df['accuracy'].min()
    y_min = min_acc - 0.01
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    plt.plot(tree_acc_df['n_trees'], tree_acc_df['accuracy'], 'o-', linewidth=2, markersize=8)

    # Add data points with values
    for x, y in zip(tree_acc_df['n_trees'], tree_acc_df['accuracy']):
        plt.text(x, y + 0.001, f'{y:.4f}', ha='center')
    
    plt.xticks(tree_acc_df['n_trees'], tree_acc_df['n_trees'])  

    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest: Number of Trees vs Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(y_min, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trees_vs_accuracy.png', dpi=300)
    plt.close()
    
    print(f'Forest hyperparameter visualization saved to {output_dir}')

def confusion_matrix(le, matrix, true, predicted, output_dir='iot_classification/graphs', fstring='1') -> None:
    '''
    Shows and saves a confusion matrix
    @param:
        le: label encoder
        matrix: sklearn confusion matrix
        true: true labels
        predicted: predicted labels
        output_dir: directory to save the graph
        fstring: string to add to the filename
    @return:
        None
    '''
    # get the names
    device_mapping = pd.read_csv('iot_classification/corpus/list_of_devices.csv', header=None, names=['Device', 'Hash'])
    device_mapping_dict = dict(zip(device_mapping['Hash'].str.strip(), device_mapping['Device'].str.strip()))

    class_labels = le.classes_

    decoded_labels = [device_mapping_dict.get(label, label) for label in class_labels]

    # plot
    plt.figure(figsize=(20, 20))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='viridis', 
                xticklabels=decoded_labels, yticklabels=decoded_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_dir}/confusion_matrix_{fstring}.png', dpi=300)
    plt.show()

    print(f'Confusion matrix saved to {output_dir}/confusion_matrix_{fstring}.png')

if __name__ == '__main__':
    #make_tree_graph_other('iot_classification/docs/tuning/5.pklrick')
    make_forest_graph('iot_classification/docs/tuning/5.pklrick')
    make_tree_graph_errors('iot_classification/docs/tuning/5.pklrick')



'''
def make_tree_graphs(pickle_path='iot_classification/docs/tuning/1.pklrick', output_dir='iot_classification/graphs'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f'Loading tree hyperparameter data from {pickle_path}...')
    with open(pickle_path, 'rb') as f:
        save_obj = pickle.load(f)
    
    # convert tree data to DataFrame
    tree_data = save_obj.trees
    df = pd.DataFrame(tree_data, columns=['max_depth', 'min_node', 'feature_count', 'unused', 'accuracy'])
    # drop the column that is used for processing in the main classify script
    df = df.drop('unused', axis=1)
    
    # Replace None values with 200 for the graph to work
    df = df.fillna(200)
    
    # Handle potential duplicate entries by taking the mean accuracy
    df = df.groupby(['max_depth', 'min_node', 'feature_count'], as_index=False)['accuracy'].mean()

    df_high = pd.DataFrame(tree_data, columns=['max_depth', 'min_node', 'feature_count', 'unused', 'accuracy'])
    df_high = df_high[df_high['accuracy'] >= 0.98]
    
    # Heatmaps weren't a good choice
    # Create heatmap for max_depth vs min_node
    print('Creating max_depth vs min_node heatmap...')
    # Group by max_depth and min_node
    depth_node_df = df.pivot_table(
        index='max_depth', 
        columns='min_node', 
        values='accuracy',
        aggfunc='mean'
    )

    min_acc = df['accuracy'].min()
    max_acc = df['accuracy'].max()

    plt.figure(figsize=(12, 8))
    sns.heatmap(depth_node_df, cmap='viridis', annot=True, fmt='.4f', vmin=0.9, vmax=max_acc)
    plt.title('Accuracy: Max Depth vs Min Node')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/max_depth_vs_min_node.png', dpi=300)
    plt.close()
    
    # Create heatmap for max_depth vs feature_count
    print('Creating max_depth vs feature_count heatmap...')
    depth_feature_df = df.pivot_table(
        index='max_depth', 
        columns='feature_count', 
        values='accuracy',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(depth_feature_df, cmap='viridis', annot=True, fmt='.4f', vmin=0.9, vmax=max_acc)
    plt.title('Accuracy: Max Depth vs Feature Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/max_depth_vs_feature_count.png', dpi=300)
    plt.close()
    
    # Create heatmap for min_node vs feature_count
    print('Creating min_node vs feature_count heatmap...')
    node_feature_df = df.pivot_table(
        index='min_node', 
        columns='feature_count', 
        values='accuracy',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(node_feature_df, cmap='viridis', annot=True, fmt='.4f', vmin=0.8, vmax=max_acc)
    plt.title('Accuracy: Min Node vs Feature Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/min_node_vs_feature_count.png', dpi=300)
    plt.close()
    
    print(f'Tree hyperparameter visualizations saved to {output_dir}')

    fig = px.parallel_coordinates(
        df, 
        color="accuracy",
        dimensions=['max_depth', 'min_node', 'feature_count', 'accuracy'],
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Decision Tree Hyperparameter Tuning Results",
        range_color=[df['accuracy'].min(), df['accuracy'].max()]
    )
    
    # Update layout for better readability
    fig.update_layout(
        font=dict(size=12),
        coloraxis_colorbar=dict(title="Accuracy"),
    )

    print(f'Saving filtered parallel coordinates html to {output_dir}...')
    fig.write_html(f'{output_dir}/tree_hyperparameters.html')    

    fig = px.parallel_coordinates(
        df_high, 
        color="accuracy",
        dimensions=['max_depth', 'min_node', 'feature_count', 'accuracy'],
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Decision Tree Hyperparameter Tuning Results (High Accuracy)",
        range_color=[0.98, df['accuracy'].max()]
    )
    
    # Update layout for better readability
    fig.update_layout(
        font=dict(size=12),
        coloraxis_colorbar=dict(title="Accuracy"),
    )

    print(f'Saving filtered parallel coordinates html to {output_dir}...')
    fig.write_html(f'{output_dir}/tree_hyperparameters_high_accuracy.html')   
    ''' 