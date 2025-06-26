import numpy as np
from typing import Dict, List, Tuple
from traders import (
    Trader, Portfolio, MomentumTrader, MeanReversionTrader, 
    ValueTrader, FixedStochasticTrader, AdaptiveStochasticTrader
)

class EvolvingTrader(Trader):
    """Base class for traders that can switch strategies based on performance"""
    
    def __init__(self, trader_id: int, initial_cash: float, 
                 initial_strategy: str, switching_threshold: int = 5,
                 performance_window: int = 20):
        super().__init__(trader_id, initial_cash)
        
        self.current_strategy_name = initial_strategy
        self.switching_threshold = switching_threshold  # Consecutive losses before switching
        self.performance_window = performance_window
        self.consecutive_losses = 0
        self.last_portfolio_value = initial_cash
        
        # Adaptive stochastic traders cannot evolve/switch strategies
        self.can_evolve = initial_strategy != 'adaptive_stochastic'
        
        # Track performance by strategy globally (shared across all evolving traders)
        # Exclude adaptive_stochastic from global tracking since it's not switchable
        if not hasattr(EvolvingTrader, 'global_strategy_performance'):
            EvolvingTrader.global_strategy_performance = {
                'momentum': [],
                'mean_reversion': [],
                'value': [],
                'fixed_stochastic': []
                # Note: adaptive_stochastic is excluded
            }
        
        # Create strategy instances
        # Adaptive stochastic traders only have their own strategy
        if initial_strategy == 'adaptive_stochastic':
            self.strategies = {
                'adaptive_stochastic': AdaptiveStochasticTrader(trader_id, initial_cash)
            }
        else:
            # Other traders can switch between all strategies except adaptive_stochastic
            self.strategies = {
                'momentum': MomentumTrader(trader_id, initial_cash),
                'mean_reversion': MeanReversionTrader(trader_id, initial_cash),
                'value': ValueTrader(trader_id, initial_cash),
                'fixed_stochastic': FixedStochasticTrader(trader_id, initial_cash)
                # Note: adaptive_stochastic is excluded from switchable strategies
            }
        
        # Share portfolio across strategies
        for strategy in self.strategies.values():
            strategy.portfolio = self.portfolio
        
        self.current_strategy = self.strategies[initial_strategy]
    
    def update_performance_tracking(self, market_prices: Dict[str, float]):
        """Track performance and decide if strategy switch is needed"""
        current_value = self.portfolio.get_value(market_prices)
        period_return = (current_value / self.last_portfolio_value) - 1
        
        # Only track performance globally for switchable strategies
        if self.can_evolve and self.current_strategy_name in EvolvingTrader.global_strategy_performance:
            EvolvingTrader.global_strategy_performance[self.current_strategy_name].append(period_return)
            
            # Keep only recent performance
            if len(EvolvingTrader.global_strategy_performance[self.current_strategy_name]) > self.performance_window:
                EvolvingTrader.global_strategy_performance[self.current_strategy_name].pop(0)
        
        # Track consecutive losses for strategy switching (only for evolving traders)
        if self.can_evolve:
            if period_return < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Switch strategy if too many consecutive losses
            if self.consecutive_losses >= self.switching_threshold:
                self.switch_to_best_strategy()
                self.consecutive_losses = 0
        
        # Update adaptive stochastic weights if using that strategy
        if self.current_strategy_name == 'adaptive_stochastic':
            self.current_strategy.update_weights(market_prices)
        
        self.last_portfolio_value = current_value
    
    def switch_to_best_strategy(self):
        """Switch to the best performing strategy based on recent global performance"""
        # Only evolving traders can switch strategies
        if not self.can_evolve:
            return
        
        avg_performance = {}
        
        # Only consider switchable strategies (excludes adaptive_stochastic)
        for strategy_name, returns in EvolvingTrader.global_strategy_performance.items():
            if returns:
                # Calculate average return and Sharpe-like metric
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 1.0
                # Simple Sharpe approximation (higher is better)
                performance_score = avg_return / (std_return + 0.001)  # Avoid division by zero
                avg_performance[strategy_name] = performance_score
            else:
                avg_performance[strategy_name] = 0
        
        # Find best strategy among available options
        if avg_performance:
            best_strategy = max(avg_performance, key=avg_performance.get)
            
            # Only switch if it's different and significantly better
            current_performance = avg_performance.get(self.current_strategy_name, 0)
            if (best_strategy != self.current_strategy_name and 
                avg_performance[best_strategy] > current_performance * 1.1):
                
                self.current_strategy_name = best_strategy
                self.current_strategy = self.strategies[best_strategy]
                print(f"Trader {self.trader_id} switched to {best_strategy} strategy")
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        """Use current strategy to decide trades"""
        return self.current_strategy.decide_trades(market_state)

def create_evolving_traders(num_traders: int, initial_cash: float) -> List[Trader]:
    """Create a mix of evolving traders"""
    traders = []
    
    # Initial strategy distribution - can start with any strategy
    initial_strategies = ['momentum', 'mean_reversion', 'value', 'fixed_stochastic', 'adaptive_stochastic']
    
    for i in range(num_traders):
        # Distribute initial strategies evenly
        initial_strategy = initial_strategies[i % len(initial_strategies)]
        traders.append(EvolvingTrader(i, initial_cash, initial_strategy))
    
    return traders