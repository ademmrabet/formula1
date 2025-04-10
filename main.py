import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import modules from project
from src.data_preparation import load_data, preprocess_data, save_processed_data, create_features_for_modeling
from src.data_exploration import explore_data, visualize_team_performance, plot_points_distribution, analyze_dominance_periods
from src.descriptive_model import perform_clustering, visualize_clusters
from src.Predictive_model import prepare_prediction_features, train_prediction_model, evaluate_model
from src.utils import create_directories, save_dataframe, load_dataframe, configure_plots, print_section_header

def main():
    # Create necessary directories
    create_directories()
    
    # Configure plotting
    configure_plots()
    
    # Step 1: Load and preprocess data
    print_section_header("Step 1: Data Loading and Preprocessing")
    
    raw_data_path = 'data/ConstructorStandings.csv'
    
    try:
        df_raw = load_data(raw_data_path)
        print(f"Loaded raw data with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns")
        
        # Perform exploratory data analysis on raw data
        print("\nExploratory analysis of raw data:")
        explore_data(df_raw)
        
        # Preprocess the data
        df_processed = preprocess_data(df_raw)
        print(f"\nPreprocessed data: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
        
        # Save processed data
        save_processed_data(df_processed, 'data/processed_constructor_data.csv')
        
    except Exception as e:
        print(f"Error in data loading/preprocessing: {e}")
        return
    
    # Step 2: Data Visualization
    print_section_header("Step 2: Data Visualization")
    
    try:
        # Visualize team performance over time
        visualize_team_performance(df_raw, top_teams=8)
        
        # Plot points distribution
        try:
            plot_points_distribution(df_processed)
        except Exception as e:
            print(f"Error in points distribution plotting: {e}")
            
        # Analyze dominance periods
        dominance_df = analyze_dominance_periods(df_raw)
        print("\nPeriods of Team Dominance (3+ consecutive championships):")
        print(dominance_df)
        
        # Save dominance analysis
        save_dataframe(dominance_df, 'team_dominance_periods', directory='data')
        
    except Exception as e:
        print(f"Error in data visualization: {e}")
    
    # Step 3: Descriptive Modeling (Clustering)
    print_section_header("Step 3: Descriptive Modeling - Clustering Analysis")
    
    try:
        # Perform clustering to identify team performance groups
        clustered_df = perform_clustering(df_processed, n_clusters=4)
        
        # Save clustered data
        save_dataframe(clustered_df, 'clustered_constructor_data', directory='data')
        
        # Visualize clusters
        visualize_clusters(clustered_df, df_processed)
        
        # Print performance group distribution
        print("\nPerformance Group Distribution:")
        perf_distribution = clustered_df['PerformanceGroup'].value_counts()
        print(perf_distribution)
        
    except Exception as e:
        print(f"Error in descriptive modeling: {e}")
    
    # Step 4: Predictive Modeling
    print_section_header("Step 4: Predictive Modeling - Position Prediction")
    
    try:
        if 'PriorChampionships' not in df_processed.columns:
            df_processed = df_processed.sort_values(['Constructor', 'Season'])
            df_processed['IsChampion'] = (df_processed['Position'] == 1).astype(int)
            df_processed['PriorChampionships'] = (
            df_processed.groupby('Constructor')['IsChampion']
                .cumsum()
                .shift()
                .fillna(0)
                .astype(int)
    )

        # Create features for modeling
        model_features = create_features_for_modeling(df_processed)
        
        # Train prediction model
        model, X_test, y_test = train_prediction_model(model_features)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
    except Exception as e:
        print(f"Error in predictive modeling: {e}")
    
    print_section_header("Analysis Complete")
    print("All analysis steps have been completed. Results and visualizations are saved in the 'assets' and 'data' directories.")

if __name__ == "__main__":
    main()