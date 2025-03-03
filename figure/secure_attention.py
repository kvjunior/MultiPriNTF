import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Set style parameters to match academic paper standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'mathtext.fontset': 'stix'
})

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Color scheme
query_color = '#0072B2'  # Blue
key_color = '#009E73'    # Green
value_color = '#D55E00'  # Orange
noise_color = '#CC79A7'  # Pink

# Standard Attention Mechanism (Left Panel)
def create_standard_attention(ax):
    # Create a graph for standard attention
    G = nx.DiGraph()
    
    # Add nodes
    query_nodes = ['Q1', 'Q2', 'Q3']
    key_nodes = ['K1', 'K2', 'K3']
    value_nodes = ['V1', 'V2', 'V3']
    
    # Add nodes with positions
    pos = {}
    for i, node in enumerate(query_nodes):
        G.add_node(node, type='query')
        pos[node] = (0, -i)
    
    for i, node in enumerate(key_nodes):
        G.add_node(node, type='key')
        pos[node] = (1, -i)
    
    for i, node in enumerate(value_nodes):
        G.add_node(node, type='value')
        pos[node] = (2, -i)
    
    # Add edges with varying thickness based on attention weights
    attention_weights = [0.7, 0.3, 0.1]
    for i, (q, k) in enumerate(zip(query_nodes, key_nodes)):
        G.add_edge(q, k, weight=attention_weights[i])
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes(data=True):
        if node[1]['type'] == 'query':
            node_colors.append(query_color)
            node_sizes.append(500)
        elif node[1]['type'] == 'key':
            node_colors.append(key_color)
            node_sizes.append(500)
        else:
            node_colors.append(value_color)
            node_sizes.append(500)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax1)
    
    # Draw edges with varying thickness
    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                width=d['weight']*5, 
                                alpha=0.6, 
                                ax=ax1)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax1)
    
    # Customize the plot
    ax1.set_title('(a) Standard Attention Mechanism', fontweight='bold')
    ax1.text(1, -3.5, 'Raw Matrices Accessible', 
             horizontalalignment='center', 
             fontsize=8, 
             color='red', 
             alpha=0.7)
    ax1.axis('off')

# Secure Attention Mechanism (Right Panel)
def create_secure_attention(ax):
    # Create a graph for secure attention
    G = nx.DiGraph()
    
    # Add nodes
    query_nodes = ['Q1', 'Q2', 'Q3']
    key_nodes = ['K1', 'K2', 'K3']
    value_nodes = ['V1', 'V2', 'V3']
    noise_nodes = ['N1', 'N2', 'N3']
    
    # Add nodes with positions
    pos = {}
    for i, node in enumerate(query_nodes):
        G.add_node(node, type='query')
        pos[node] = (0, -i)
    
    for i, node in enumerate(key_nodes):
        G.add_node(node, type='key')
        pos[node] = (1, -i)
    
    for i, node in enumerate(value_nodes):
        G.add_node(node, type='value')
        pos[node] = (2, -i)
    
    for i, node in enumerate(noise_nodes):
        G.add_node(node, type='noise')
        pos[node] = (1.5, -i-3.5)
    
    # Add edges with varying thickness based on attention weights
    attention_weights = [0.6, 0.25, 0.15]
    for i, (q, k) in enumerate(zip(query_nodes, key_nodes)):
        G.add_edge(q, k, weight=attention_weights[i])
    
    # Add noise edges
    for i, k in enumerate(key_nodes):
        G.add_edge(noise_nodes[i], k, weight=0.2, style='dashed')
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes(data=True):
        if node[1]['type'] == 'query':
            node_colors.append(query_color)
            node_sizes.append(500)
        elif node[1]['type'] == 'key':
            node_colors.append(key_color)
            node_sizes.append(500)
        elif node[1]['type'] == 'value':
            node_colors.append(value_color)
            node_sizes.append(500)
        else:
            node_colors.append(noise_color)
            node_sizes.append(300)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax2)
    
    # Draw edges with varying thickness
    for (u, v, d) in G.edges(data=True):
        if d.get('style', 'solid') == 'dashed':
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                    width=d['weight']*3, 
                                    alpha=0.5, 
                                    style='dashed',
                                    edge_color=noise_color,
                                    ax=ax2)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                    width=d['weight']*5, 
                                    alpha=0.6, 
                                    ax=ax2)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax2)
    
    # Customize the plot
    ax2.set_title('(b) Secure Attention Mechanism', fontweight='bold')
    
    # Add noise injection explanation
    ax2.text(1.5, -4.5, 'Differentially Private\nNoise Injection', 
             horizontalalignment='center', 
             fontsize=8, 
             color=noise_color, 
             alpha=0.8)
    
    ax2.axis('off')

# Create the visualizations
create_standard_attention(ax1)
create_secure_attention(ax2)

# Adjust layout and add overall title
plt.tight_layout(pad=2.0)
fig.suptitle('Secure Attention Mechanism with Privacy Preservation', 
             fontweight='bold', 
             fontsize=12, 
             y=0.98)

# Save the figure
plt.savefig('figures/secure_attention.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/secure_attention.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Secure attention mechanism visualization generated successfully.")