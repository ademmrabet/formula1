"""
Utility functions for the F1 Constructor Analysis project
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def create_directories():
    """
    Create necessary directories for the project
    """
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory")
    
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")

def save_dataframe(df, filename, directory='data'):
    """
    Save a DataFrame to a CSV file
    
    Args:
        df: DataFrame to save
        filename: Name of the file (without extension)
        directory: Directory to save the file in
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f"{filename}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved DataFrame to {filepath}")

def load_dataframe(filename, directory='data'):
    """
    Load a DataFrame from a CSV file
    
    Args:
        filename: Name of the file (without extension)
        directory: Directory where the file is located
    
    Returns:
        DataFrame with the loaded data
    """
    filepath = os.path.join(directory, f"{filename}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    return pd.read_csv(filepath)

def configure_plots():
    """
    Configure matplotlib plot settings for better visualization
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
def print_section_header(title):
    """
    Print a formatted section header for console output
    
    Args:
        title: Title of the section
    """
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")