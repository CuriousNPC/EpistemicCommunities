import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_participation_by_cost(results_path: str, output_dir: str):
    """
    Analyze how participation rates vary with cost structure.
    Expects CSV with columns: time_step, cost, participation_rate
    """
    # Read data
    df = pd.read_csv(results_path)
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate average participation rate for each cost level
    cost_participation = df.groupby('cost')['participation_rate'].mean().reset_index()
    
    # Plot cost vs participation (static relationship)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cost_participation, x='cost', y='participation_rate')
    plt.title('Average Participation Rate by Cost Level')
    plt.xlabel('Cost')
    plt.ylabel('Participation Rate')
    plt.savefig(Path(output_dir) / 'cost_participation_static.png')
    plt.close()
    
    # Analyze stability of participation rates over time for each cost level
    plt.figure(figsize=(12, 6))
    for cost in sorted(df['cost'].unique()):
        cost_data = df[df['cost'] == cost]
        plt.plot(cost_data['time_step'], cost_data['participation_rate'], 
                label=f'Cost={cost}')
    
    plt.title('Participation Rate Over Time by Cost Level')
    plt.xlabel('Time Step')
    plt.ylabel('Participation Rate')
    plt.legend()
    plt.savefig(Path(output_dir) / 'cost_participation_temporal.png')
    plt.close()
    
    # Generate statistics
    stats = {
        'participation_by_cost': cost_participation.to_dict(),
        'variability': df.groupby('cost')['participation_rate'].std().to_dict()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    results_path = "full_results.csv"  # Path to your results CSV
    output_dir = "participation_analysis"
    stats = analyze_participation_by_cost(results_path, output_dir)
    print("Statistics:", stats)
