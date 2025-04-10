import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def explore_data(df):
    #explore the basic characteristics of the dataset
    print("\nData Overview:")
    print(f"shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())


    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    #Count unique Constructors
    print(f"\nTotal unique constructors: {df['Team'].nunique()}")

    #Distribution of points across seasons
    print("\nPoints distribution by year:")
    yearly_stats = df.groupby('Year')['PTS'].agg(['mean', 'min','max']).reset_index()
    print(yearly_stats.head(10)) #to show only the first 10 years

def visualize_team_performance(df, top_teams=10):
    # Visualize the performance of top F1 constructors over time
    plt.figure(figsize=(14, 8))

    # Identify top teams by number of champions
    champion_counts = df[df['PTS'] == 1]['Team'].value_counts()
    top_constructors = champion_counts.head(top_teams).index.tolist()

    # Create filtered dataset with just the top teams
    top_teams_data = df[df['Team'].isin(top_constructors)]

    # Check if there is any data available for the top teams
    if top_teams_data.empty:
        print("No data available for the selected top teams.")
        return  # Exit the function if there's no data to plot

    # Plot positions over time for top teams
    for team in top_constructors:
        team_data = top_teams_data[top_teams_data['Team'] == team]
        
        # Check if the team has data before plotting
        if not team_data.empty:
            plt.plot(team_data['Year'], team_data['PTS'], 'o-', label=team, alpha=0.7)

    # Format the plot
    plt.gca().invert_yaxis()  # Invert Y-axis so position 1 is at the top
    plt.title('Constructor Championship Positions Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Championship Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, -0.5), ncol=2)
    plt.tight_layout()

    # Set y-axis to show only integer values for positions
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the figure
    plt.savefig('assets/team_performance_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_points_distribution(df):
    if 'Year' not in df.columns and 'Season' in df.columns:
        df['Year'] = df['Season']  # fallback in case column was renamed


    if 'Position' in df.columns and 'Pos' not in df.columns:
        df['Pos'] = df['Position']  # fallback mapping

    # Ensure 'Pos' is numeric
    df['Pos'] = pd.to_numeric(df['Pos'], errors='coerce')

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Boxplot by era
    df['Era'] = pd.cut(df['Year'],
                       bins=[1957, 1970, 1980, 1990, 2000, 2010, 2021],
                       labels=['1958-1970', '1971-1980', '1981-1990',
                               '1991-2000', '2001-2010', '2011-2020'])
    sns.boxplot(x='Era', y='Points', data=df, ax=axes[0])
    axes[0].set_title('Points Distribution by Era')
    axes[0].set_xlabel('Era')
    axes[0].set_ylabel('Constructor Points')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Champion vs Runner-up scatter
    df['IsChampion'] = (df['Pos'] == 1).astype(int)
    champion_data = df[df['IsChampion'] == 1]
    runner_up_data = df[df['Pos'] == 2]

    axes[1].scatter(champion_data['Year'], champion_data['Points'],
                    color='gold', label='Champions', s=80, alpha=0.7, edgecolor='black')
    axes[1].scatter(runner_up_data['Year'], runner_up_data['Points'],
                    color='silver', label='Runner-ups', s=60, alpha=0.6, edgecolor='black')

    axes[1].set_title('Champion vs Runner-up Points Over Time')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Points')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/points_distribution.png', dpi=300)
    plt.close()

    print("Points distribution visualization saved to 'assets/points_distribution.png'")


def analyze_dominance_periods(df):
    """
     Analyze periods of team dominance in F1 history
     Calculate the number of consecutive championships for each constructor
    """

    #Identify Consecutive Championships by the same team

    champions = df[df['Pos'] == 1].sort_values('Year')

    #Calculate Consecutive Years
    champions['PrevYear'] = champions['Year'].shift(1)
    champions['PrevTeam'] = champions['Team'].shift(1)
    champions['ConsecutiveYear'] = (champions['Year'] - champions['PrevYear'] == 1)
    champions['SameTeamAsPrev'] = (champions['Team'] == champions['PrevTeam'])

    #Identify dominance periods (3+ consecutive Championships)
    champions['DominanceStart'] = (champions['SameTeamAsPrev'] & champions['ConsecutiveYear'] &
                                   (~champions['SameTeamAsPrev'].shift(1) | ~champions['ConsecutiveYear'].shift(1)))
    

    #Group consecutive championships
    champions['DominancePeriod'] = champions['DominanceStart'].cumsum()

    #Count consecutive championships in each dominance period
    dominance_df = champions.groupby(['Team', 'DominancePeriod']).size().reset_index(name='ConsecutiveChampionships')


    #Filter for true dominance (3+ consecutive championships)
    dominance_df = dominance_df[dominance_df['ConsecutiveChampionships'] >= 3]

    return dominance_df