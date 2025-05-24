import os
import json
from simulation import MarketSimulation, ExperimentRunner

def create_distributions(total_traders: int = 100):
    """Create different trader distributions for experiments"""
    
    distributions = [
        # Experiment 1: Equal distribution
        {
            'momentum': 20,
            'mean_reversion': 20,
            'value': 20,
            'fixed_stochastic': 20,
            'adaptive_stochastic': 20
        },
        
        # Experiment 2: Deterministic heavy
        {
            'momentum': 30,
            'mean_reversion': 30,
            'value': 30,
            'fixed_stochastic': 5,
            'adaptive_stochastic': 5
        },
        
        # Experiment 3: Stochastic heavy
        {
            'momentum': 10,
            'mean_reversion': 10,
            'value': 10,
            'fixed_stochastic': 35,
            'adaptive_stochastic': 35
        },
        
        # Experiment 4: Adaptive heavy
        {
            'momentum': 15,
            'mean_reversion': 15,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 40
        },
        
        # Experiment 5: Momentum dominated
        {
            'momentum': 40,
            'mean_reversion': 15,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 15
        },
        
        # Experiment 6: Mean reversion dominated
        {
            'momentum': 15,
            'mean_reversion': 40,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 15
        },
        
        # Experiment 7: No stochastic
        {
            'momentum': 33,
            'mean_reversion': 33,
            'value': 34,
            'fixed_stochastic': 0,
            'adaptive_stochastic': 0
        },
        
        # Experiment 8: Only stochastic
        {
            'momentum': 0,
            'mean_reversion': 0,
            'value': 0,
            'fixed_stochastic': 50,
            'adaptive_stochastic': 50
        }
    ]
    
    return distributions

def print_experiment_overview(distributions):
    """Print overview of all experiments"""
    print("="*60)
    print("EXPERIMENT OVERVIEW")
    print("="*60)
    
    for i, dist in enumerate(distributions, 1):
        total = sum(dist.values())
        print(f"\nExperiment {i}:")
        print(f"  Description: {get_experiment_description(dist)}")
        print(f"  Distribution:")
        for strategy, count in dist.items():
            if count > 0:
                percentage = (count / total) * 100
                print(f"    - {strategy.replace('_', ' ').title()}: {count} traders ({percentage:.1f}%)")
    print("\n" + "="*60)

def get_experiment_description(distribution):
    """Generate a descriptive name for the experiment"""
    total = sum(distribution.values())
    
    # Find dominant strategy
    max_count = max(distribution.values())
    dominant_strategies = [k for k, v in distribution.items() if v == max_count]
    
    if len(dominant_strategies) == 1 and max_count / total > 0.4:
        return f"{dominant_strategies[0].replace('_', ' ').title()} Dominated"
    elif max_count / total <= 0.25:
        return "Equal Distribution"
    else:
        # Find major categories
        deterministic = distribution.get('momentum', 0) + distribution.get('mean_reversion', 0) + distribution.get('value', 0)
        stochastic = distribution.get('fixed_stochastic', 0) + distribution.get('adaptive_stochastic', 0)
        
        if deterministic > stochastic * 1.5:
            return "Deterministic Heavy"
        elif stochastic > deterministic * 1.5:
            return "Stochastic Heavy"
        else:
            return "Mixed Strategies"

def main():
    """Run the full experiment suite"""
    
    # Create results directory structure
    directories = ['results', 'results/data', 'results/plots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Base configuration
    config = {
        'initial_cash': 10000,
        'risk_tolerance': 0.5
    }
    
    # Save configuration
    with open('results/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Define trader distributions to test
    distributions = create_distributions()
    
    # Print experiment overview
    print_experiment_overview(distributions)
    
    print("\nNOTE: All traders can now evolve their strategies!")
    print("- Traders switch strategies after consecutive losses")
    print("- Strategy selection based on global performance tracking")
    print("- Stochastic traders adjust weights based on relative performance")
    print("\nStarting experiment suite...")
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Run experiments
    summary_df = runner.run_experiment_suite(distributions, num_runs=3)
    
    # Analyze results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    runner.analyze_equilibria(summary_df)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print("Files generated:")
    print("📊 Summary: results/experiment_summary.csv")
    print("📈 Individual plots: results/plots/experiment_X_Y_*.png")
    print("📄 Detailed data: results/data/experiment_X_Y_*.json")
    print("⚙️  Configuration: results/config.json")
    
    print(f"\n🔍 Key insights to explore:")
    print("- Strategy convergence patterns across different initial distributions")
    print("- Performance differences between evolving vs. static strategies")
    print("- Market dynamics under different trader compositions")
    print("- Equilibrium states and convergence times")
    
    print(f"\n📁 For LaTeX integration:")
    print("- Each plot is saved as a separate high-resolution PNG")
    print("- JSON files contain complete experimental data and metadata")
    print("- File naming follows experiment_X.Y pattern for easy referencing")

if __name__ == "__main__":
    main()