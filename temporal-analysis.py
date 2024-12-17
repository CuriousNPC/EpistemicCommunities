import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_update_dynamics(results_path: str, output_dir: str):
    """
    Analyze how update_lag and update_sensitivity affect consensus formation.
    Expects CSV with columns: time_step, update_lag, update_sensitivity, consensus_accuracy
    """
    # Read data
    df = pd.read_csv(results_path)
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Analyze effect of update lag
    plt.figure(figsize=(10, 6))
    for lag in sorted(df['update_lag'].unique()):
        lag_data = df[df['update_lag'] == lag]
        sns.lineplot(data=lag_data, x='time_step', y='consensus_accuracy',
                    label=f'Lag={lag}')
    
    plt.title('Consensus Formation by Update Lag')
    plt.xlabel('Time Step')
    plt.ylabel('Consensus Accuracy')
    plt.savefig(Path(output_dir) / 'update_lag_effects.png')
    plt.close()
    
    # Analyze effect of update sensitivity
    plt.figure(figsize=(10, 6))
    for sens in sorted(df['update_sensitivity'].unique()):
        sens_data = df[df['update_sensitivity'] == sens]
        sns.lineplot(data=sens_data, x='time_step', y='consensus_accuracy',
                    label=f'Sensitivity={sens}')
    
    plt.title('Consensus Formation by Update Sensitivity')
    plt.xlabel('Time Step')
    plt.ylabel('Consensus Accuracy')
    plt.savefig(Path(output_dir) / 'update_sensitivity_effects.png')
    plt.close()
    
    # Generate statistics
    stats = {
        'accuracy_by_lag': df.groupby('update_lag')['consensus_accuracy'].mean().to_dict(),
        'accuracy_by_sensitivity': df.groupby('update_sensitivity')['consensus_accuracy'].mean().to_dict(),
        'convergence_time': {
            'by_lag': df.groupby('update_lag').agg(
                {'time_step': lambda x: x[df['consensus_accuracy'] > 0.9].min()}
            ).to_dict()['time_step'],
            'by_sensitivity': df.groupby('update_sensitivity').agg(
                {'time_step': lambda x: x[df['consensus_accuracy'] > 0.9].min()}
            ).to_dict()['time_step']
        }
    }
    
    return stats

if __name__ == "__main__":
    results_path = "full_results.csv"  # Path to your results CSV
    output_dir = "temporal_analysis"
    stats = analyze_update_dynamics(results_path, output_dir)
    print("Statistics:", stats)
