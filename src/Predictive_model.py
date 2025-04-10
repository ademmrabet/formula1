import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_prediction_features(df):
    # Ensure PriorChampionships column exists
    if 'PriorChampionships' not in df.columns:
        df = df.sort_values(['Constructor', 'Season'])
        df['IsChampion'] = (df['Position'] == 1).astype(int)
        df['PriorChampionships'] = (
            df.groupby('Constructor')['IsChampion']
            .cumsum()
            .shift()
            .fillna(0)
            .astype(int)
        )

    # Group lesser teams into 'Other'
    top_teams = df['Constructor'].value_counts().head(15).index
    df['TeamCategory'] = df['Constructor'].apply(lambda x: x if x in top_teams else 'Other')

    # Define features
    features = ['Points', 'Season', 'TeamCategory', 'NormalizedPoints', 'PriorChampionships']
    X = df[features].copy()
    y = df['Position']

    return X, y


def train_prediction_model(df):

    if 'PriorChampionships' not in df.columns:
        df = df.sort_values(['Constructor', 'Season'])
        df['IsChampion'] = (df['Position'] == 1).astype(int)
        df['PriorChampionships'] = (
        df.groupby('Constructor')['IsChampion']
        .cumsum()
        .shift()
        .fillna(0)
        .astype(int)
    )

    # Prepare features and target
    X, y = prepare_prediction_features(df)

    #split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessing steps
    numeric_features = ['Points', 'NormalizedPoints', 'Season']
    categorical_features = ['TeamCategory']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create and train the model
    # Decision tree for interpretability

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=6, random_state=42))
    ])

    #Fit the model
    model.fit(X_train, y_train)
    print("Decision tree model trained successfully.")

    # Random Forest for comparison
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")

    # compare models using cross-validation
    dt_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

    print(f"\nDecision Tree RMSE (CV): {np.sqrt(-dt_scores.mean()):.4f}")
    print(f"Random Forest RMSE (CV): {np.sqrt(-rf_scores.mean()):.4f}")


    # What's the best model
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):

    #make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    #Calculate percentage of predictions within 1 position
    accurate_predictions = np.abs(y_test - y_pred) <= 1
    accuracy_within_one = np.mean(accurate_predictions) * 100
    print(f"Prediction within 1 position: {accuracy_within_one:.2f}%")

    # Visualize actual vs predicted positions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    #Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('Actual vs Predicted Constructor Positions', fontsize=14)
    plt.xlabel('Actual Position', fontsize=12)
    plt.ylabel('Predicted Position', fontsize=12)
    plt.grid(True, alpha=0.3)


    #Add text with metrics
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig('assets/position_prediction_results.png', dpi=300)
    plt.close()


    #extract the decision tree for visualization
    decision_tree = model.named_steps['regressor']
    feature_names = (
        model.named_steps['preprocessor']
        .transformers_[0][1]
        .get_feature_names_out(input_features=['Points', 'NormalizedPoints', 'Season'])
        .tolist() +
        model.named_steps['preprocessor']
        .transformers_[1][1]
        .get_feature_names_out(input_features=['TeamCategory'])
        .tolist()
    )
    
    # Create decision tree visualization
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree)