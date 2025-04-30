import processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class NetworkTrafficVisualizer:
    '''
    Class for creating visualizations for network traffic data.
    '''

    def __init__(self, filepaths = 'darknet/corpus/parts/*.csv', kwargs = {}):
        self.data = processing.Data(filepaths, kwargs).data
        self.cols = self._get_numeric_columns()

    def _get_numeric_columns(self):
        '''
        Get numeric columns from the DataFrame.
        '''
        return self.data.select_dtypes(include=[np.number]).columns.tolist() 
    
    def plot_correlation_matrix(self, n_features: int = 20):
        '''
        Plot a correlation matrix for the numeric columns in the DataFrame.
        '''
        print("Plotting correlation matrix...")

        plt.figure(figsize=(14, 12))

        # Select top N features by variance
        df = self.data.copy()[self.cols]
        v = df.var().sort_values(ascending=False)
        top_features = v.index[:n_features].to_list()

        # Calculate the correlation matrix for the top features
        corr = df[top_features].corr()

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, 
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap (Top {n_features} Features by Variance)')
        plt.tight_layout()
        plt.savefig('darknet/graphs/correlation_matrix.png')
        plt.show()

    def plot_feature_distribution(self, n_features: int = 20):
        '''
        Plot the distribution of top N features.
        '''
        print("Plotting feature distribution...")
        plt.figure(figsize=(10, 6))

        # Select top N features by variance
        df = self.data.copy()[self.cols]
        v = df.var().sort_values(ascending=False)
        top_features = v.index[:n_features].to_list()

        fig, axes = plt.subplots(n_features // 2, 2, figsize=(14, 3 * n_features // 2))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            v
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            
            # Add vertical lines for key statistics
            mean = self.data[feature].mean()
            median = self.data[feature].median()
            axes[i].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            axes[i].axvline(median, color='g', linestyle='-.', label=f'Median: {median:.2f}')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f'darknet/graphs/feature_distribution.png')
        plt.show()

    def plot_protocol_distribution(self):
        '''
        Plot the distribution of a specific protocol.
        '''
        print("Plotting protocol distribution...")
        plt.figure(figsize=(10, 6))
        
        if 'Protocol' in self.data.columns:
            protocol_counts = self.data['Protocol'].value_counts()
            
            # Map protocols to names if numeric
            protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
            if isinstance(protocol_counts.index[0], (int, np.integer)):
                protocol_counts.index = protocol_counts.index.map(
                    lambda x: f"{x} ({protocol_map.get(x, 'Unknown')})" if x in protocol_map else x
                )
            
            sns.barplot(x=protocol_counts.index, y=protocol_counts.values)
            plt.title('Protocol Distribution')
            plt.xlabel('Protocol')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'darknet/graphs/protocol_distribution.png')
            plt.show()

    def plot_network_graph(self, max_edges: int = 100):
        '''
        Plot a network graph of the data.
        '''
        print("Plotting network graph...")

        if 'Src IP' in self.data.columns and 'Dst IP' in self.data.columns:
            n = nx.DiGraph()

            connections = self.data.groupby(['Src IP', 'Dst IP']).size().reset_index(name='weight')
            connections = connections.sort_values('weight', ascending=False).head(max_edges)

            for _, row in connections.iterrows():
                n.add_edge(row['Src IP'], row['Dst IP'], weight=row['weight'])
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(n, k=0.3, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(n, pos, node_size=100, alpha=0.8, 
                                  node_color='lightblue', linewidths=0.5, edgecolors='black')
            
            # Draw edges with width based on weight
            edge_widths = [n[u][v]['weight'] / connections['weight'].max() * 3 for u, v in n.edges()]
            nx.draw_networkx_edges(n, pos, width=edge_widths, alpha=0.6, 
                                  edge_color='gray', arrowsize=15)
            
            # Draw labels
            nx.draw_networkx_labels(n, pos, font_size=8, font_family='sans-serif')
            
            plt.title('Network Traffic Graph (Top Connections)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('darknet/graphs/network_graph.png')
            plt.show()
            
            # Print statistics
            print("Network Graph Statistics:")
            print(f"Number of nodes (unique IPs): {n.number_of_nodes()}")
            print(f"Number of edges (connections): {n.number_of_edges()}")
            
            # Identify key nodes
            in_degree = dict(n.in_degree())
            out_degree = dict(n.out_degree())
            
            # Top IPs by incoming connections
            top_receivers = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop destination IPs (by incoming connections):")
            for ip, degree in top_receivers:
                print(f"{ip}: {degree} incoming connections")
                
            # Top IPs by outgoing connections
            top_senders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop source IPs (by outgoing connections):")
            for ip, degree in top_senders:
                print(f"{ip}: {degree} outgoing connections")

    def plot_port_heatmap(self):
        '''
        Plot a heatmap of the port distribution.
        '''
        if 'src port' in self.data.columns and 'dst port' in self.data.columns:
            port_ranges = [
                (0, 1023, 'Well-known'),
                (1024, 49151, 'Registered'),
                (49152, 65535, 'Dynamic/Private')
            ]
            
            # Categorize ports
            def categorize_port(port):
                for start, end, name in port_ranges:
                    if start <= port <= end:
                        return name
                return 'Unknown'
            
            # Add categories
            plot_df = self.data.copy()
            plot_df['Src Port Category'] = plot_df['src port'].apply(categorize_port)
            plot_df['Dst Port Category'] = plot_df['dst port'].apply(categorize_port)
            
            # Create cross-tabulation
            port_crosstab = pd.crosstab(
                plot_df['Src Port Category'], 
                plot_df['Dst Port Category']
            )
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(port_crosstab, annot=True, fmt='d', cmap='YlGnBu')
            plt.title('Source Port vs. Destination Port Categories')
            plt.xlabel('Destination Port Category')
            plt.ylabel('Source Port Category')
            plt.tight_layout()
            plt.savefig('darknet/graphs/port_heatmap.png')
            plt.show()
            
            # Plot top ports
            plt.figure(figsize=(12, 6))
            top_dst_ports = self.data['dst port'].value_counts().head(10)
            sns.barplot(x=top_dst_ports.index, y=top_dst_ports.values)
            plt.title('Top 10 Destination Ports')
            plt.xlabel('Port')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('darknet/graphs/top_dst_ports.png')
            plt.show()

def main():
    visualizer = NetworkTrafficVisualizer()
    # visualizer.plot_correlation_matrix(n_features=20)
    # # visualizer.plot_feature_distribution(n_features=5)
    # visualizer.plot_protocol_distribution()
    # visualizer.plot_network_graph(max_edges=100)
    visualizer.plot_port_heatmap()

if __name__ == "__main__":
    main()