import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os
from datetime import datetime

# Create a folder with a timestamp in the name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Adjust these paths as needed
output_folder = f"chess_metrics_{timestamp}"
file_path = r'chess_metrics_sft_atm_sl5_mnt100.jsonl'


os.makedirs(output_folder, exist_ok=True)

# Function to read JSONL file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Read data
try:
    data = read_jsonl(file_path)
    print(f"Successfully loaded {len(data)} records")
except FileNotFoundError:
    raise FileNotFoundError("Results file not found.")

# Create DataFrame
df = pd.DataFrame(data)

# Calculate legal move fraction
df['legal_move_fraction'] = df['num_legal_moves_suggested'] / df['total_moves']

# Extract model name without path for better visualization
df['model_short_name'] = df['model_name'].apply(lambda x: x.split('/')[-1])

# Set up the visualization
sns.set_theme(style="whitegrid")

# Metrics to plot
metrics = [
    ('move_accuracy', 'Move Accuracy'),
    ('legal_move_fraction', 'Legal Move Fraction'),
    ('avg_move_quality', 'Average Move Quality')
]

# Plot each metric in a separate figure
for metric, title in metrics:
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(
        x='model_short_name',
        y=metric,
        hue='prompt_method',
        data=df,
        palette="viridis"
    )
    
    # Enhance readability
    plt.title(title, fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Prompt Method')
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='bottom', 
            fontsize=8, rotation=0
        )
    
    # Save each plot separately
    plot_path = os.path.join(output_folder, f'{metric}_bar_chart.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

print(f"Plots saved in folder: {output_folder}")