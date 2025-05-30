import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime
import os

from market import Market, Stock
from traders import (
    MomentumTrader, MeanReversionTrader, ValueTrader,
    FixedStochasticTrader, AdaptiveStochasticTrader
)
from evolutionary_traders import EvolvingTrader

class MarketSimulation:
    """Main simulation engine for the agent-based financial market"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.market = self._initialize_market()
        self.traders = []
        self.results = {
            'trader_performance': defaultdict(list),
            'strategy_proportions': defaultdict(list),
            'market_metrics': defaultdict(list),
            'convergence_data': []
        }
        
    def _initialize_market(self) -> Market:
        """Initialize market with stocks"""
        stocks = []
        
        # Stock A: High volatility, trending
        stocks.append(Stock(
            symbol='A',
            current_price=100.0,
            fundamental_value=100.0,
            volatility=0.02,  # Reduced from 0.03
            drift=0.0005,     # Reduced from 0.001
            price_impact=0.1,  # Reduced from 0.2
            characteristic='trending'
        ))
        
        # Stock B: Mean-reverting, low volatility
        stocks.append(Stock(
            symbol='B',
            current_price=50.0,
            fundamental_value=50.0,
            volatility=0.01,
            drift=0.0,
            price_impact=0.1,  # Reduced from 0.15
            characteristic='mean_reverting'
        ))
        
        # Stock C: Random walk
        stocks.append(Stock(
            symbol='C',
            current_price=75.0,
            fundamental_value=75.0,
            volatility=0.015,  # Reduced from 0.02
            drift=0.0,
            price_impact=0.08,  # Reduced from 0.1
            characteristic='random_walk'
        ))
        
        # Stock D: Cyclical patterns
        stocks.append(Stock(
            symbol='D',
            current_price=25.0,
            fundamental_value=25.0,
            volatility=0.01,   # Reduced from 0.015
            drift=0.0,
            price_impact=0.1,  # Reduced from 0.12
            characteristic='cyclical'
        ))
        
        return Market(stocks, self.config['initial_cash'])
    
    def _initialize_traders(self, trader_distribution: Dict[str, int]) -> List:
        """Initialize traders based on distribution"""
        traders = []
        trader_id = 0
        
        # Import at the top of the method
        from traders import Trader
        
        # Create a market maker first to ensure liquidity
        class MarketMaker(Trader):
            """Simple market maker to provide liquidity"""
            def __init__(self, trader_id, initial_cash, market):
                super().__init__(trader_id, initial_cash * 2)  # Give more capital
                # Start with inventory in all stocks
                market_prices = {s: stock.current_price for s, stock in market.stocks.items()}
                for symbol, price in market_prices.items():
                    quantity = int((initial_cash * 0.4) / (price * len(market_prices)))
                    self.portfolio.holdings[symbol] = quantity
                    self.portfolio.cash -= quantity * price
            
            def decide_trades(self, market_state):
                trades = []
                for symbol, state in market_state.items():
                    # Always willing to buy and sell small amounts
                    if np.random.random() < 0.3:  # 30% chance to provide liquidity
                        if self.portfolio.cash > state['price'] * 5:
                            trades.append((symbol, 'buy', 5))
                        if self.portfolio.holdings.get(symbol, 0) > 5:
                            trades.append((symbol, 'sell', 5))
                return trades
        
        # Add market maker
        market_maker = MarketMaker(trader_id, self.config['initial_cash'], self.market)
        traders.append(market_maker)
        trader_id += 1
        
        # All traders are now evolving traders that can switch between any strategy
        for _ in range(trader_distribution.get('momentum', 0)):
            initial_strategy = 'momentum'
            traders.append(EvolvingTrader(trader_id, self.config['initial_cash'], initial_strategy))
            trader_id += 1
            
        for _ in range(trader_distribution.get('mean_reversion', 0)):
            initial_strategy = 'mean_reversion'
            traders.append(EvolvingTrader(trader_id, self.config['initial_cash'], initial_strategy))
            trader_id += 1
            
        for _ in range(trader_distribution.get('value', 0)):
            initial_strategy = 'value'
            traders.append(EvolvingTrader(trader_id, self.config['initial_cash'], initial_strategy))
            trader_id += 1
        
        for _ in range(trader_distribution.get('fixed_stochastic', 0)):
            initial_strategy = 'fixed_stochastic'
            traders.append(EvolvingTrader(trader_id, self.config['initial_cash'], initial_strategy))
            trader_id += 1
        
        for _ in range(trader_distribution.get('adaptive_stochastic', 0)):
            initial_strategy = 'adaptive_stochastic'
            traders.append(EvolvingTrader(trader_id, self.config['initial_cash'], initial_strategy))
            trader_id += 1
        
        # Give some traders initial holdings to enable selling
        np.random.seed(42)  # For reproducibility
        market_prices = {s: stock.current_price for s, stock in self.market.stocks.items()}
        
        for i, trader in enumerate(traders[1:], 1):  # Skip market maker
            if i % 3 == 0:  # Every third trader starts with some holdings
                # Randomly assign holdings in 1-2 stocks
                symbols = list(market_prices.keys())
                np.random.shuffle(symbols)
                for symbol in symbols[:2]:
                    price = market_prices[symbol]
                    max_quantity = int((trader.portfolio.cash * 0.3) / price)
                    if max_quantity > 0:
                        quantity = np.random.randint(1, max_quantity + 1)
                        trader.portfolio.holdings[symbol] = quantity
                        trader.portfolio.cash -= quantity * price
            
        return traders
    
    def run_single_period(self):
        """Run one trading period"""
        market_state = self.market.get_market_state()
        
        # Each trader decides on trades
        for trader in self.traders:
            trades = trader.decide_trades(market_state)
            
            # Submit orders to market
            for symbol, action, quantity in trades:
                if action == 'buy':
                    self.market.submit_order(trader.trader_id, symbol, 'buy', quantity)
                elif action == 'sell':
                    self.market.submit_order(trader.trader_id, symbol, 'sell', quantity)
        
        # Execute trades
        executed_trades = self.market.execute_trades()
        
        # Update trader portfolios based on executed trades
        for symbol, trades in executed_trades.items():
            for trade in trades:
                # Update buyer
                buyer = self.traders[trade['buyer_id']]
                buyer.portfolio.execute_trade(symbol, trade['quantity'], 
                                            trade['price'], 'buy')
                
                # Update seller
                seller = self.traders[trade['seller_id']]
                seller.portfolio.execute_trade(symbol, trade['quantity'], 
                                             trade['price'], 'sell')
        
        # Update adaptive traders and evolutionary mechanisms
        market_prices = {symbol: stock.current_price 
                        for symbol, stock in self.market.stocks.items()}
        
        for trader in self.traders:
            # Update all evolving traders
            if isinstance(trader, EvolvingTrader):
                trader.update_performance_tracking(market_prices)
            # Keep support for non-evolving traders if needed
            elif isinstance(trader, AdaptiveStochasticTrader):
                trader.update_weights(market_prices)
            
            trader.update_performance(market_prices)
    
    def calculate_strategy_proportions(self) -> Dict[str, float]:
        """Calculate current proportion of each strategy type"""
        strategy_counts = defaultdict(int)
        total_traders = len(self.traders) - 1  # Exclude market maker
        
        for trader in self.traders[1:]:  # Skip market maker
            if isinstance(trader, EvolvingTrader):
                # Count current strategy being used
                strategy_counts[trader.current_strategy_name] += 1
            elif isinstance(trader, MomentumTrader):
                strategy_counts['momentum'] += 1
            elif isinstance(trader, MeanReversionTrader):
                strategy_counts['mean_reversion'] += 1
            elif isinstance(trader, ValueTrader):
                strategy_counts['value'] += 1
            elif isinstance(trader, FixedStochasticTrader):
                strategy_counts['fixed_stochastic'] += 1
            elif isinstance(trader, AdaptiveStochasticTrader):
                strategy_counts['adaptive_stochastic'] += 1
        
        return {strategy: count / total_traders 
                for strategy, count in strategy_counts.items()}
    
    def calculate_performance_by_strategy(self) -> Dict[str, Dict[str, float]]:
        """Calculate average performance metrics by strategy type"""
        performance = defaultdict(lambda: {'returns': [], 'sharpe': []})
        market_prices = {symbol: stock.current_price 
                        for symbol, stock in self.market.stocks.items()}
        
        for trader in self.traders[1:]:  # Skip market maker
            returns = trader.portfolio.get_returns(market_prices)
            
            # Calculate Sharpe ratio (simplified)
            if trader.performance_history:
                avg_return = np.mean(trader.performance_history)
                std_return = np.std(trader.performance_history)
                sharpe = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe = 0
            
            # Categorize by strategy
            if isinstance(trader, EvolvingTrader):
                # Use current strategy
                performance[trader.current_strategy_name]['returns'].append(returns)
                performance[trader.current_strategy_name]['sharpe'].append(sharpe)
            elif isinstance(trader, MomentumTrader):
                performance['momentum']['returns'].append(returns)
                performance['momentum']['sharpe'].append(sharpe)
            elif isinstance(trader, MeanReversionTrader):
                performance['mean_reversion']['returns'].append(returns)
                performance['mean_reversion']['sharpe'].append(sharpe)
            elif isinstance(trader, ValueTrader):
                performance['value']['returns'].append(returns)
                performance['value']['sharpe'].append(sharpe)
            elif isinstance(trader, FixedStochasticTrader):
                performance['fixed_stochastic']['returns'].append(returns)
                performance['fixed_stochastic']['sharpe'].append(sharpe)
            elif isinstance(trader, AdaptiveStochasticTrader):
                performance['adaptive_stochastic']['returns'].append(returns)
                performance['adaptive_stochastic']['sharpe'].append(sharpe)
        
        # Calculate averages
        avg_performance = {}
        for strategy, metrics in performance.items():
            avg_performance[strategy] = {
                'avg_returns': np.mean(metrics['returns']) if metrics['returns'] else 0,
                'std_returns': np.std(metrics['returns']) if metrics['returns'] else 0,
                'avg_sharpe': np.mean(metrics['sharpe']) if metrics['sharpe'] else 0
            }
        
        return avg_performance
    
    def check_convergence(self, window: int = 50, threshold: float = 0.01) -> bool:
        """Check if strategy proportions have converged"""
        if len(self.results['strategy_proportions']['momentum']) < window:
            return False
        
        converged = True
        for strategy in ['momentum', 'mean_reversion', 'value', 
                        'fixed_stochastic', 'adaptive_stochastic']:
            recent_props = self.results['strategy_proportions'][strategy][-window:]
            if np.std(recent_props) > threshold:
                converged = False
                break
        
        return converged
    
    def run_simulation(self, trader_distribution: Dict[str, int], 
                      max_periods: int = 1000, check_convergence: bool = True) -> Dict:
        """Run complete simulation"""
        self.traders = self._initialize_traders(trader_distribution)
        
        print(f"Starting simulation with {len(self.traders)} traders...")
        print(f"Distribution: {trader_distribution}")
        
        converged = False
        convergence_period = None
        
        for period in range(max_periods):
            # Run trading period
            self.run_single_period()
            
            # Record results every 10 periods
            if period % 10 == 0:
                # Strategy proportions
                proportions = self.calculate_strategy_proportions()
                for strategy, prop in proportions.items():
                    self.results['strategy_proportions'][strategy].append(prop)
                
                # Performance metrics
                performance = self.calculate_performance_by_strategy()
                for strategy, metrics in performance.items():
                    self.results['trader_performance'][strategy].append(metrics)
                
                # Market metrics
                for symbol, stock in self.market.stocks.items():
                    self.results['market_metrics'][f'{symbol}_price'].append(
                        stock.current_price
                    )
                    self.results['market_metrics'][f'{symbol}_volatility'].append(
                        self.market.get_volatility(symbol)
                    )
                
                # Check convergence
                if check_convergence and not converged and period > 100:
                    if self.check_convergence():
                        converged = True
                        convergence_period = period
                        print(f"Convergence detected at period {period}")
                        if period < max_periods - 200:  # Continue for a bit after convergence
                            continue
                        else:
                            break
            
            # Progress update
            if period % 100 == 0:
                print(f"Period {period}/{max_periods}")
        
        # Final results
        final_results = {
            'converged': converged,
            'convergence_period': convergence_period,
            'final_proportions': self.calculate_strategy_proportions(),
            'final_performance': self.calculate_performance_by_strategy(),
            'results_history': self.results
        }
        
        return final_results
    
    def plot_results(self, results: Dict, experiment_name: str, save_dir: str = "results/plots"):
        """Generate separate plots for each metric"""
        
        # Create plots directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        
        # 1. Strategy proportions over time
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in ['momentum', 'mean_reversion', 'value',
                         'fixed_stochastic', 'adaptive_stochastic']:
            if strategy in results['results_history']['strategy_proportions']:
                ax.plot(
                    results['results_history']['strategy_proportions'][strategy],
                    label=strategy.replace('_', ' ').title(),
                    linewidth=2
                )
        ax.set_xlabel('Time (periods/10)', fontsize=12)
        ax.set_ylabel('Strategy Proportion', fontsize=12)
        ax.set_title(f'{experiment_name} - Strategy Proportions Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_strategy_proportions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Average return over time by strategy
        fig, ax = plt.subplots(figsize=(10, 6))
        perf_hist = results['results_history']['trader_performance']
        for strategy, metrics_list in perf_hist.items():
            ret_series = [m['avg_returns'] for m in metrics_list]
            ax.plot(ret_series, label=strategy.replace('_', ' ').title(), linewidth=2)
        ax.set_xlabel('Time (periods/10)', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.set_title(f'{experiment_name} - Average Returns Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_returns_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Final average returns (bar chart)
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = []
        avg_returns = []
        for strategy, perf in results['final_performance'].items():
            strategies.append(strategy.replace('_', ' ').title())
            avg_returns.append(perf['avg_returns'])
        
        bars = ax.bar(strategies, avg_returns, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#1D5F3F'])
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Average Returns', fontsize=12)
        ax.set_title(f'{experiment_name} - Final Performance by Strategy', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_final_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Stock prices over time
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, symbol in enumerate(['A', 'B', 'C', 'D']):
            key = f'{symbol}_price'
            if key in results['results_history']['market_metrics']:
                ax.plot(results['results_history']['market_metrics'][key],
                       label=f'Stock {symbol}', color=colors[i], linewidth=2)
        ax.set_xlabel('Time (periods/10)', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'{experiment_name} - Stock Price Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_stock_prices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Return distribution (violin plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        performance_data = []
        strategy_labels = []
        for trader in self.traders[1:]:  # Skip market maker
            price_map = {s: stk.current_price for s, stk in self.market.stocks.items()}
            r = trader.portfolio.get_returns(price_map)
            strategy_labels.append(
                getattr(trader, 'current_strategy_name', 
                       trader.__class__.__name__.replace('Trader','').lower()).replace('_', ' ').title()
            )
            performance_data.append(r)
        
        df = pd.DataFrame({'Strategy': strategy_labels, 'Returns': performance_data})
        sns.violinplot(x='Strategy', y='Returns', data=df, ax=ax, palette='Set2')
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Returns', fontsize=12)
        ax.set_title(f'{experiment_name} - Return Distribution by Strategy', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{experiment_name}_return_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved for {experiment_name}")

class ExperimentRunner:
    """Run multiple experiments with different configurations"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.experiment_results = []
        
    def run_experiment_suite(self, distributions: List[Dict[str, int]], 
                           num_runs: int = 3) -> pd.DataFrame:
        """Run multiple experiments with different trader distributions"""
        
        # Create directories
        os.makedirs('results/data', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        
        for dist_idx, distribution in enumerate(distributions):
            print(f"\nExperiment {dist_idx + 1}/{len(distributions)}")
            
            # Create experiment name
            strategy_mix = "_".join(f"{k[:2]}{v}" for k, v in distribution.items() if v > 0)
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                
                # Create and run simulation
                sim = MarketSimulation(self.base_config)
                results = sim.run_simulation(distribution)
                
                # Create experiment identifier
                experiment_name = f"experiment_{dist_idx+1}_{run+1}"
                
                # Record results
                experiment_data = {
                    'experiment_id': experiment_name,
                    'distribution': distribution,
                    'converged': results['converged'],
                    'convergence_period': results['convergence_period'],
                    **{f'final_prop_{k}': v for k, v in results['final_proportions'].items()},
                    **{f'final_perf_{k}_returns': v['avg_returns'] 
                       for k, v in results['final_performance'].items()},
                    **{f'final_perf_{k}_sharpe': v['avg_sharpe'] 
                       for k, v in results['final_performance'].items()}
                }
                
                self.experiment_results.append(experiment_data)
                
                # Save detailed results with initial state
                detailed_results = {
                    'experiment_info': {
                        'experiment_id': experiment_name,
                        'experiment_name': f"Experiment {dist_idx+1}.{run+1}",
                        'description': self._get_experiment_description(distribution),
                        'initial_distribution': distribution,
                        'config': self.base_config
                    },
                    'simulation_results': {
                        'converged': results['converged'],
                        'convergence_period': results['convergence_period'],
                        'final_proportions': results['final_proportions'],
                        'final_performance': results['final_performance']
                    },
                    'time_series_data': {
                        'strategy_proportions': dict(results['results_history']['strategy_proportions']),
                        'trader_performance': dict(results['results_history']['trader_performance']),
                        'market_metrics': dict(results['results_history']['market_metrics'])
                    }
                }
                
                json_filename = f"results/data/{experiment_name}_{strategy_mix}.json"
                self.save_results(detailed_results, json_filename)
                
                # Generate separate plots
                sim.plot_results(results, experiment_name)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.experiment_results)
        summary_df.to_csv('results/experiment_summary.csv', index=False)
        
        return summary_df
    
    def _get_experiment_description(self, distribution: Dict[str, int]) -> str:
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
    
    def save_results(self, results: Dict, filename: str):
        """Save detailed results to JSON file"""
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def analyze_equilibria(self, summary_df: pd.DataFrame):
        """Analyze equilibrium patterns across experiments"""
        
        # Group by initial distribution pattern
        print("\n=== Equilibrium Analysis ===")
        
        # Check for convergence patterns
        convergence_rate = summary_df['converged'].mean()
        print(f"Overall convergence rate: {convergence_rate:.2%}")
        
        # Average convergence time
        converged_df = summary_df[summary_df['converged']]
        if not converged_df.empty:
            avg_convergence_time = converged_df['convergence_period'].mean()
            print(f"Average convergence time: {avg_convergence_time:.0f} periods")
        
        # Equilibrium strategy proportions
        print("\nEquilibrium Strategy Proportions:")
        prop_columns = [col for col in summary_df.columns if col.startswith('final_prop_')]
        for col in prop_columns:
            strategy = col.replace('final_prop_', '')
            avg_prop = summary_df[col].mean()
            std_prop = summary_df[col].std()
            print(f"  {strategy}: {avg_prop:.3f} ± {std_prop:.3f}")
        
        # Performance analysis
        print("\nAverage Performance by Strategy:")
        perf_columns = [col for col in summary_df.columns if 'final_perf_' in col and '_returns' in col]
        for col in perf_columns:
            strategy = col.replace('final_perf_', '').replace('_returns', '')
            avg_perf = summary_df[col].mean()
            std_perf = summary_df[col].std()
            print(f"  {strategy}: {avg_perf:.3f} ± {std_perf:.3f}")
        
        return convergence_rate