import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict
from itertools import product
import os
from datetime import datetime
from tqdm import tqdm

from abm_model import InformationDynamicsABM, UtilityType

class MainAnalysis:
    def __init__(self, n_iterations=100, base_output_dir="experiment_results"):
        self.n_iterations = n_iterations
        self.base_output_dir = base_output_dir
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_theme(style="whitegrid")
        
    def run_analysis(self):
        """Run full analysis pipeline"""
        print("Starting data collection and simulation...")
        raw_df, output_dir = self.run_iteration_sweep()
        
        print("\nCreating visualization directory...")
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        self.create_visualizations(raw_df, viz_dir)
        
        print("\nGenerating summary statistics...")
        self.generate_summary_stats(raw_df, output_dir)
        
        return output_dir

    def run_iteration_sweep(self):
        """Run parameter sweep with multiple iterations"""
        # Parameter ranges
        p_values = [0.3, 0.5, 0.7]  # Information quality
        cost_values = [0.1, 0.3, 0.5, 0.7]  # Participation costs
        network_densities = [0.1, 0.5, 0.8]  # Network densities
        update_sensitivities = [0.7, 1.0, 1.3]  # Update sensitivities
        update_lags = [1, 5, 10]  # Update lags

        demographic_configs = {
            "balanced": {
                UtilityType.TRUTH_SEEKER: 0.33,
                UtilityType.FREE_RIDER: 0.33,
                UtilityType.MIXED: 0.34
            },
            "mixed_heavy": {
                UtilityType.TRUTH_SEEKER: 0.2,
                UtilityType.FREE_RIDER: 0.2,
                UtilityType.MIXED: 0.6
            },
            "truth_seeker_heavy": {
                UtilityType.TRUTH_SEEKER: 0.6,
                UtilityType.FREE_RIDER: 0.2,
                UtilityType.MIXED: 0.2
            }
        }

        all_results = []
        param_combinations = list(product(
            p_values, cost_values, network_densities,
            update_sensitivities, update_lags, demographic_configs.items()
        ))

        with tqdm(total=len(param_combinations) * self.n_iterations,
                  desc="Running simulations") as pbar:
            
            for params in param_combinations:
                p, cost, density, sensitivity, lag, (demo_name, demo_dist) = params
                
                for iteration in range(self.n_iterations):
                    model = InformationDynamicsABM(
                        n_agents=100,
                        true_state=1.0,
                        p=p,
                        cost=cost,
                        demographic_distribution=demo_dist,
                        network_density=density,
                        update_sensitivity=sensitivity,
                        k_star=10,
                        update_lag=lag
                    )

                    # Track consensus over time
                    for step in range(20):
                        model.step()
                        state = {
                            'iteration': iteration,
                            'time_step': step,
                            'p': p,
                            'cost': cost,
                            'network_density': density,
                            'update_sensitivity': sensitivity,
                            'update_lag': lag,
                            'demographic_config': demo_name,
                            'consensus': model.get_consensus(),
                            'consensus_accuracy': float(model.get_consensus() == model.true_state)
                            if model.get_consensus() is not None else None,
                            'participation_rate': len([a for a in model.agents.values()
                                                       if a.expressed_belief is not None]) / model.n_agents,
                            'opinion_diversity': model.get_opinion_diversity()
                        }
                        all_results.append(state)
                    pbar.update(1)

        raw_df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.base_output_dir, f'run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        raw_df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
        
        return raw_df, output_dir

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
        plt.figure(figsize=(15, 5))
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

        # Combined effects dashboard
        self._plot_combined_dashboard(df, viz_dir)

    def _plot_combined_dashboard(self, df: pd.DataFrame, viz_dir: str):
        """Create combined effects dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Combined Effects Dashboard')

        # Network density vs consensus accuracy
        sns.lineplot(
            data=df,
            x='network_density',
            y='consensus_accuracy',
            hue='p',
            ax=axes[0,0],
            errorbar=('ci', 95)
        )
        axes[0,0].set_title('Network Effects')
        
        # Information quality vs consensus accuracy
        sns.lineplot(
            data=df,
            x='p',
            y='consensus_accuracy',
            hue='demographic_config',
            ax=axes[0,1],
            errorbar=('ci', 95)
        )
        axes[0,1].set_title('Information Quality Effects')
        
        # Demographic comparison
        sns.boxplot(
            data=df,
            x='demographic_config',
            y='consensus_accuracy',
            ax=axes[1,0]
        )
        axes[1,0].set_title('Demographic Effects')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Network density x information quality interaction
        pivot_data = df.groupby(['network_density', 'p'])['consensus_accuracy'].mean().reset_index()
        pivot_table = pivot_data.pivot('network_density', 'p', 'consensus_accuracy')
        sns.heatmap(pivot_table, ax=axes[1,1], annot=True, fmt='.2f', cmap='viridis')
        axes[1,1].set_title('Density x Info Quality Interaction')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'combined_dashboard.png'))
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
    analyzer = MainAnalysis(n_iterations=100)
    output_dir = analyzer.run_analysis()
    print(f"\nAnalysis complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()