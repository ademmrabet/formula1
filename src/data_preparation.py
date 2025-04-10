"""
Data preparation functions for F1 Constructor Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the raw constructor standings data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        DataFrame with the loaded data.
    """
    try:
        #If reading the CSV directly
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        # If this is the first run and we need to create the file from the problem statement
        print("Raw data file not found. Creating from the provided data...")
        
        # This is where you would paste the data from the problem statement
        # For this example, we'll assume the data is already in the right place
        
        # If you're running this for the first time, you might need to save the data 
        # from the problem statement into raw_data.csv first
        raise FileNotFoundError(f"Please ensure {file_path} exists with the F1 Constructor data")
    
def preprocess_data(df):
    # Clean and preprocess the raw data
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Remove unnecessary columns
    if 'Unnamed: 0' in processed_df.columns:
        processed_df = processed_df.drop('Unnamed: 0', axis=1)

    # Rename columns for clarity
    processed_df = processed_df.rename(columns={
        'Pos': 'Position',
        'Team': 'Constructor',
        'PTS': 'Points',
        'Year': 'Season',
    })

    # Handle non-numeric positions (like 'EX')
    # Convert to numeric, coercing errors to NaN
    processed_df['Position'] = pd.to_numeric(processed_df['Position'], errors='coerce')
    
    # Replace NaN values (from 'EX') with a meaningful numerical value
    # For excluded/disqualified teams, using a value like 20 (or higher than max position)
    processed_df['Position'] = processed_df['Position'].fillna(20)
    
    # Convert to integers after handling NaNs
    processed_df['Position'] = processed_df['Position'].astype(int)
    
    # Continue with the rest of your preprocessing
    processed_df['Points'] = processed_df['Points'].astype(float)
    processed_df['Season'] = processed_df['Season'].astype(int)

    # Create era categories
    processed_df['Era'] = pd.cut(
        processed_df['Season'],
        bins=[1957, 1970, 1980, 1990, 2000, 2010, 2021],
        labels=['1958-1970', '1971-1980', '1981-1990', '1991-2000', '2001-2010', '2011-2020']
    )

    # Extract engine manufacturer from constructor name where possible
    processed_df['Engine'] = processed_df['Constructor'].str.extract(r'(\w+)$')

    # Identify the dominant teams
    champion_mask = processed_df['Position'] == 1
    processed_df['IsChampion'] = champion_mask.astype(int)

    # Create normalized points (points as percentage of max points that season)
    season_max_points = processed_df.groupby('Season')['Points'].transform('max')
    processed_df['NormalizedPoints'] = processed_df['Points'] / season_max_points * 100

    # Handle any missing values
    processed_df = processed_df.fillna({'Engine': 'Unknown'})

    return processed_df

def save_processed_data(df, file_path):
    """
    Save the processed data to a CSV file.
    Args:
        df (DataFrame): Processed DataFrame to save.
        file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"Processed data with {df.shape[0]} rows and {df.shape[1]} columns saved ")

def create_features_for_modeling(df):
    #create features for machine learning
    #create copy to avoid modifying the original
    features_df = df.copy()

    #create dummy variables for categorical features
    features_df = pd.get_dummies(features_df, columns=['Era'], drop_first=True)


    #create additional features
    #Number of championships won by constructors prior to the current season
    constructor_champions = features_df.sort_values(['Constructor', 'Season'])
    constructor_champions['PriorChampionships'] = constructor_champions.groupby('Constructor')['IsChampion'].cumsum().shift(1).fillna(0)


    #Previous season position (if available)
    constructor_history = features_df.sort_values(['Constructor', 'Season'])
    constructor_history['PrevSeasonPosition'] = constructor_history.groupby('Constructor')['Position'].shift(1)

    #Create features for modeling
    model_features = constructor_history[['Season', 'Constructor', 'Position', 'Points', 
                                          'NormalizedPoints', 'IsChampion', 'PriorChampionships',
                                          'PrevSeasonPosition']]
    
    #fill missing values for previous season position
    model_features['PrevSeasonPosition'] = model_features['PrevSeasonPosition'].fillna(20) #Assume P20 if no history ( P20 = last place) 

    #Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Points', 'NormalizedPoints', 'PriorChampionships', 'PrevSeasonPosition']
    model_features[numerical_cols] = scaler.fit_transform(model_features[numerical_cols])

    return model_features