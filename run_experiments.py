import os
import json
from simulation import MarketSimulation, ExperimentRunner

def create_distributions(total_traders: int = 100):
    """Create different trader distributions for experiments"""
    
    distributions = [
        # Equal distribution
        {
            'momentum': 20,
            'mean_reversion': 20,
            'value': 20,
            'fixed_stochastic': 20,
            'adaptive_stochastic': 20
        },
        
        # Deterministic heavy
        {
            'momentum': 30,
            'mean_reversion': 30,
            'value': 30,
            'fixed_stochastic': 5,
            'adaptive_stochastic': 5
        },
        
        # Stochastic heavy
        {
            'momentum': 10,
            'mean_reversion': 10,
            'value': 10,
            'fixed_stochastic': 35,
            'adaptive_stochastic': 35
        },
        
        # Adaptive heavy
        {
            'momentum': 15,
            'mean_reversion': 15,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 40
        },
        
        # Momentum dominated
        {
            'momentum': 40,
            'mean_reversion': 15,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 15
        },
        
        # Mean reversion dominated
        {
            'momentum': 15,
            'mean_reversion': 40,
            'value': 15,
            'fixed_stochastic': 15,
            'adaptive_stochastic': 15
        },
        
        # No stochastic
        {
            'momentum': 33,
            'mean_reversion': 33,
            'value': 34,
            'fixed_stochastic': 0,
            'adaptive_stochastic': 0
        },
        
        # Only stochastic
        {
            'momentum': 0,
            'mean_reversion': 0,
            'value': 0,
            'fixed_stochastic': 50,
            'adaptive_stochastic': 50
        }
    ]
    
    return distributions

def main():
    """Run the full experiment suite"""
    
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Base configuration
    config = {
        'initial_cash': 10000,
        'risk_tolerance': 0.5
    }
    
    # Save configuration
    with open('results/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Note: Traders can now evolve their strategies based on performance!")
    print("Deterministic traders switch strategies after consecutive losses.")
    print("Stochastic traders adjust their strategy weights based on relative performance.\n")
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Define trader distributions to test
    distributions = create_distributions()
    
    # Run experiments
    print("Starting experiment suite...")
    summary_df = runner.run_experiment_suite(distributions, num_runs=3)
    
    # Analyze results
    print("\n" + "="*50)
    runner.analyze_equilibria(summary_df)
    
    print("\nExperiments complete! Results saved in 'results' directory.")
    print("Summary statistics saved to 'results/experiment_summary.csv'")
    print("\nKey insights:")
    print("- Check if strategy proportions converge to different equilibria")
    print("- Compare performance of evolving vs fixed strategies")
    print("- Analyze which strategies dominate in different market conditions")

if __name__ == "__main__":
    main()