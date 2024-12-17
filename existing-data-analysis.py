import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

class ExistingDataAnalysis:
    def __init__(self, input_path: str, base_output_dir="analysis_results"):
        self.input_path = input_path
        self.base_output_dir = base_output_dir
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_theme(style="whitegrid")
        
    def run_analysis(self):
        """Run analysis pipeline on existing data"""
        print("Loading data...")
        df = pd.read_csv(self.input_path)
        
        print("\nCreating visualization directory...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.base_output_dir, f'analysis_{timestamp}')
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        self.create_visualizations(df, viz_dir)
        
        print("\nGenerating summary statistics...")
        self.generate_summary_stats(df, output_dir)
        
        return output_dir

    def create_visualizations(self, df: pd.DataFrame, viz_dir: str):
        """Create core visualizations"""
        # Cost effects on participation
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x='cost',
            y='participation_rate',
            hue='p',
            style='demographic_config',
            errorbar=('ci', 95)
        )
        plt.title('Cost Effect on Participation Rate')
        plt.xlabel('Cost')
        plt.ylabel('Participation Rate')
        plt.savefig(os.path.join(viz_dir, 'cost_effects.png'))
        plt.close()

        # Network density effects
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x='network_density',
            y='consensus_accuracy',
            hue='p',
            style='demographic_config',
            errorbar=('ci', 95)
        )
        plt.title('Network Density Effect on Consensus Accuracy')
        plt.xlabel('Network Density')
        plt.ylabel('Consensus Accuracy')
        plt.savefig(os.path.join(viz_dir, 'network_effects.png'))
        plt.close()

        # Information quality effects
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x='p',
            y='consensus_accuracy',
            hue='demographic_config',
            style='network_density',
            errorbar=('ci', 95)
        )
        plt.title('Information Quality Effect on Consensus Accuracy')
        plt.xlabel('Information Quality (p)')
        plt.ylabel('Consensus Accuracy')
        plt.savefig(os.path.join(viz_dir, 'info_quality_effects.png'))
        plt.close()

        # Cost vs participation by environment type
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Bad info environment (p = 0.3)
        sns.lineplot(
            data=df[df['p'] == 0.3],
            x='cost',
            y='participation_rate',
            hue='demographic_config',
            errorbar=('ci', 95),
            ax=ax1
        )
        ax1.set_title('Bad Info Environment (p=0.3)')
        
        # Neutral environment (p = 0.5)
        sns.lineplot(
            data=df[df['p'] == 0.5],
            x='cost',
            y='participation_rate',
            hue='demographic_config',
            errorbar=('ci', 95),
            ax=ax2
        )
        ax2.set_title('Neutral Environment (p=0.5)')
        
        # Good info environment (p = 0.7)
        sns.lineplot(
            data=df[df['p'] == 0.7],
            x='cost',
            y='participation_rate',
            hue='demographic_config',
            errorbar=('ci', 95),
            ax=ax3
        )
        ax3.set_title('Good Info Environment (p=0.7)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'cost_effects_by_environment.png'))
        plt.close()

    def generate_summary_stats(self, df: pd.DataFrame, output_dir: str):
        """Generate core summary statistics"""
        summary = {
            'cost_effects': {
                'mean_participation_by_cost': df.groupby('cost')['participation_rate'].mean().to_dict(),
                'participation_by_cost_and_p': df.groupby(['cost', 'p'])['participation_rate'].mean().to_dict()
            },
            'overall_consensus': {
                'mean_accuracy': df['consensus_accuracy'].mean(),
                'std_accuracy': df['consensus_accuracy'].std()
            },
            'by_info_quality': df.groupby('p').agg({
                'consensus_accuracy': ['mean', 'std']
            }).to_dict(),
            'by_network': df.groupby('network_density').agg({
                'consensus_accuracy': ['mean', 'std']
            }).to_dict(),
            'by_demographic': df.groupby('demographic_config').agg({
                'consensus_accuracy': ['mean', 'std']
            }).to_dict()
        }
        
        with open(os.path.join(output_dir, 'summary_stats.txt'), 'w') as f:
            for category, stats in summary.items():
                f.write(f"\n{category.upper()}:\n")
                f.write(str(stats))
                f.write("\n" + "="*50 + "\n")

def main():
    # Specify path to your existing results file
    results_path = "full_results.csv"  # Change this to your file path
    
    print("Starting analysis of existing data...")
    analyzer = ExistingDataAnalysis(results_path)
    output_dir = analyzer.run_analysis()
    print(f"\nAnalysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()