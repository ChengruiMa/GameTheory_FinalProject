import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import random

class Portfolio:
    """Manages trader's holdings and cash"""
    
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.holdings = {}  # {symbol: quantity}
        self.initial_value = initial_cash
        self.trade_history = []
        
    def can_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if trader has enough cash to buy"""
        return self.cash >= quantity * price
    
    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if trader has enough shares to sell"""
        return self.holdings.get(symbol, 0) >= quantity
    
    def execute_trade(self, symbol: str, quantity: int, price: float, trade_type: str):
        """Execute a trade and update portfolio"""
        if trade_type == 'buy':
            if self.can_buy(symbol, quantity, price):
                self.cash -= quantity * price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                self.trade_history.append({
                    'type': 'buy',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price
                })
        elif trade_type == 'sell':
            if self.can_sell(symbol, quantity):
                self.cash += quantity * price
                self.holdings[symbol] -= quantity
                if self.holdings[symbol] == 0:
                    del self.holdings[symbol]
                self.trade_history.append({
                    'type': 'sell',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price
                })
    
    def get_value(self, market_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        value = self.cash
        for symbol, quantity in self.holdings.items():
            if symbol in market_prices:
                value += quantity * market_prices[symbol]
        return value
    
    def get_returns(self, market_prices: Dict[str, float]) -> float:
        """Calculate portfolio returns"""
        current_value = self.get_value(market_prices)
        return (current_value / self.initial_value) - 1

class Trader(ABC):
    """Abstract base class for all traders"""
    
    def __init__(self, trader_id: int, initial_cash: float, risk_tolerance: float = 0.5):
        self.trader_id = trader_id
        self.portfolio = Portfolio(initial_cash)
        self.risk_tolerance = risk_tolerance  # 0 = risk averse, 1 = risk seeking
        self.performance_history = []
        
    @abstractmethod
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        """Decide what trades to make based on market state
        Returns: List of (symbol, action, quantity) tuples"""
        pass
    
    def calculate_position_size(self, confidence: float, available_cash: float) -> float:
        """Calculate position size based on confidence and risk tolerance"""
        # Kelly-like criterion adjusted by risk tolerance
        base_fraction = min(confidence * self.risk_tolerance, 0.15)  # Cap at 15% of portfolio
        return available_cash * base_fraction
    
    def update_performance(self, market_prices: Dict[str, float]):
        """Update performance tracking"""
        returns = self.portfolio.get_returns(market_prices)
        self.performance_history.append(returns)

# Deterministic Strategies

class MomentumTrader(Trader):
    """Follows price momentum - buys rising stocks, sells falling ones"""
    
    def __init__(self, trader_id: int, initial_cash: float, 
                 lookback_period: int = 5, threshold: float = 0.02):
        super().__init__(trader_id, initial_cash)
        self.lookback_period = lookback_period
        self.threshold = threshold
        # Start with some random initial holdings
        self.initialize_holdings = True
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        trades = []
        
        # Initialize with some holdings on first trade
        if self.initialize_holdings and self.portfolio.cash > 1000:
            for symbol in list(market_state.keys())[:2]:  # Buy first 2 stocks
                price = market_state[symbol]['price']
                quantity = int((self.portfolio.cash * 0.2) / price)
                if quantity > 0:
                    trades.append((symbol, 'buy', quantity))
            self.initialize_holdings = False
            return trades
        
        for symbol, state in market_state.items():
            # Use returns_5 if available, otherwise calculate from price history
            if f'returns_{self.lookback_period}' in state:
                returns = state[f'returns_{self.lookback_period}']
            else:
                returns = state.get('returns_5', 0)
            
            current_price = state['price']
            current_holding = self.portfolio.holdings.get(symbol, 0)
            
            # Strong upward momentum - buy
            if returns > self.threshold:
                confidence = min(abs(returns) / self.threshold, 2.0) / 2.0
                position_size = self.calculate_position_size(confidence, self.portfolio.cash)
                quantity = int(position_size / current_price)
                if quantity > 0 and self.portfolio.can_buy(symbol, quantity, current_price):
                    trades.append((symbol, 'buy', quantity))
            
            # Strong downward momentum - sell
            elif returns < -self.threshold and current_holding > 0:
                quantity = min(int(current_holding * 0.5), current_holding)  # Sell half
                if quantity > 0:
                    trades.append((symbol, 'sell', quantity))
            
            # Neutral momentum - small random trades for liquidity
            elif abs(returns) < self.threshold * 0.5:
                if np.random.random() < 0.1:  # 10% chance
                    if current_holding > 20 and np.random.random() < 0.5:
                        # Small sell
                        quantity = min(10, current_holding // 4)
                        trades.append((symbol, 'sell', quantity))
                    elif self.portfolio.cash > current_price * 20:
                        # Small buy
                        trades.append((symbol, 'buy', 10))
        
        return trades

class MeanReversionTrader(Trader):
    """Trades on assumption that prices revert to mean"""
    
    def __init__(self, trader_id: int, initial_cash: float, 
                 lookback_period: int = 20, z_score_threshold: float = 2.0):
        super().__init__(trader_id, initial_cash)
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.price_history = {}
    
    def update_price_history(self, symbol: str, price: float):
        """Maintain rolling price history"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback_period:
            self.price_history[symbol].pop(0)
    
    def calculate_z_score(self, symbol: str, current_price: float) -> float:
        """Calculate z-score of current price"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.0
        
        prices = np.array(self.price_history[symbol])
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return 0.0
        
        return (current_price - mean) / std
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        trades = []
        
        for symbol, state in market_state.items():
            current_price = state['price']
            self.update_price_history(symbol, current_price)
            
            z_score = self.calculate_z_score(symbol, current_price)
            
            # Price too high - sell
            if z_score > self.z_score_threshold and self.portfolio.holdings.get(symbol, 0) > 0:
                quantity = int(self.portfolio.holdings[symbol] * 0.5)
                if quantity > 0:
                    trades.append((symbol, 'sell', quantity))
            
            # Price too low - buy
            elif z_score < -self.z_score_threshold:
                confidence = min(abs(z_score) / self.z_score_threshold, 2.0) / 2.0
                position_size = self.calculate_position_size(confidence, self.portfolio.cash)
                quantity = int(position_size / current_price)
                if quantity > 0:
                    trades.append((symbol, 'buy', quantity))
        
        return trades

class ValueTrader(Trader):
    """Trades based on fundamental value estimates"""
    
    def __init__(self, trader_id: int, initial_cash: float, 
                 value_threshold: float = 0.1):
        super().__init__(trader_id, initial_cash)
        self.value_threshold = value_threshold
        self.fundamental_estimates = {}  # Trader's estimate of fundamental values
    
    def estimate_fundamental_value(self, symbol: str, market_state: Dict) -> float:
        """Estimate fundamental value based on various factors"""
        state = market_state[symbol]
        current_price = state['price']
        
        # Simple estimation based on historical average and volatility
        if symbol not in self.fundamental_estimates:
            self.fundamental_estimates[symbol] = current_price
        else:
            # Slowly adjust estimate based on new information
            volatility = state['volatility']
            adjustment = np.random.normal(0, volatility * 0.1)
            self.fundamental_estimates[symbol] *= (1 + adjustment)
        
        return self.fundamental_estimates[symbol]
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        trades = []
        
        for symbol, state in market_state.items():
            current_price = state['price']
            fundamental_value = self.estimate_fundamental_value(symbol, market_state)
            
            price_to_value = current_price / fundamental_value
            
            # Undervalued - buy
            if price_to_value < (1 - self.value_threshold):
                confidence = min((1 - price_to_value) / self.value_threshold, 1.0)
                position_size = self.calculate_position_size(confidence, self.portfolio.cash)
                quantity = int(position_size / current_price)
                if quantity > 0:
                    trades.append((symbol, 'buy', quantity))
            
            # Overvalued - sell
            elif price_to_value > (1 + self.value_threshold) and self.portfolio.holdings.get(symbol, 0) > 0:
                quantity = int(self.portfolio.holdings[symbol] * 0.5)
                if quantity > 0:
                    trades.append((symbol, 'sell', quantity))
        
        return trades

# Stochastic Strategies

class FixedStochasticTrader(Trader):
    """Uses fixed probability matrix to randomly select from deterministic strategies"""
    
    def __init__(self, trader_id: int, initial_cash: float, strategy_weights: Dict[str, float] = None):
        super().__init__(trader_id, initial_cash)
        
        # Default equal weights if not specified

        if strategy_weights is None:
            strategies = ['momentum', 'mean_reversion', 'value']
            random_weights = np.random.dirichlet(np.ones(len(strategies)))
            strategy_weights = dict(zip(strategies, random_weights))

        self.strategy_weights = strategy_weights
        self.strategies = {
            'momentum': MomentumTrader(trader_id, initial_cash),
            'mean_reversion': MeanReversionTrader(trader_id, initial_cash),
            'value': ValueTrader(trader_id, initial_cash)
        }
        
        # Share portfolio across strategies
        for strategy in self.strategies.values():
            strategy.portfolio = self.portfolio
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        # Randomly select strategy based on weights
        strategy_name = np.random.choice(
            list(self.strategy_weights.keys()),
            p=list(self.strategy_weights.values())
        )
        
        strategy = self.strategies[strategy_name]
        return strategy.decide_trades(market_state)

class AdaptiveStochasticTrader(Trader):
    """Dynamically adjusts strategy weights based on recent performance"""
    
    def __init__(self, trader_id: int, initial_cash: float, 
                 learning_rate: float = 0.1, performance_window: int = 10):
        super().__init__(trader_id, initial_cash)

        self.learning_rate = learning_rate
        self.performance_window = performance_window

        # Initialize with random weights summing to 1
        strategies = ['momentum', 'mean_reversion', 'value']
        random_weights = np.random.dirichlet(np.ones(len(strategies)))
        self.strategy_weights = dict(zip(strategies, random_weights))

        self.strategies = {
            'momentum': MomentumTrader(trader_id, initial_cash),
            'mean_reversion': MeanReversionTrader(trader_id, initial_cash),
            'value': ValueTrader(trader_id, initial_cash)
        }
        
        # Track performance by strategy
        self.strategy_performance = {name: [] for name in self.strategies}
        self.last_strategy = None
        self.last_portfolio_value = initial_cash
        
        # Share portfolio across strategies
        for strategy in self.strategies.values():
            strategy.portfolio = self.portfolio
    
    def update_weights(self, market_prices: Dict[str, float]):
        """Update strategy weights based on recent performance"""
        if self.last_strategy is not None:
            # Calculate return from last trade
            current_value = self.portfolio.get_value(market_prices)
            trade_return = (current_value / self.last_portfolio_value) - 1
            self.strategy_performance[self.last_strategy].append(trade_return)
            
            # Keep only recent performance
            if len(self.strategy_performance[self.last_strategy]) > self.performance_window:
                self.strategy_performance[self.last_strategy].pop(0)
            
            # Update weights using exponential weighting of returns
            avg_returns = {}
            for name, returns in self.strategy_performance.items():
                if returns:
                    avg_returns[name] = np.mean(returns)
                else:
                    avg_returns[name] = 0
            
            # Softmax transformation to get probabilities
            if avg_returns:
                max_return = max(avg_returns.values())
                exp_returns = {name: np.exp((ret - max_return) / 0.1) 
                             for name, ret in avg_returns.items()}
                total_exp = sum(exp_returns.values())
                
                for name in self.strategy_weights:
                    if name in exp_returns:
                        new_weight = exp_returns[name] / total_exp
                        # Smooth update
                        self.strategy_weights[name] = (
                            (1 - self.learning_rate) * self.strategy_weights[name] +
                            self.learning_rate * new_weight
                        )
                
                # Normalize weights
                total_weight = sum(self.strategy_weights.values())
                self.strategy_weights = {name: w/total_weight 
                                       for name, w in self.strategy_weights.items()}
            
            self.last_portfolio_value = current_value
    
    def decide_trades(self, market_state: Dict) -> List[Tuple[str, str, int]]:
        # Select strategy based on current weights
        strategy_name = np.random.choice(
            list(self.strategy_weights.keys()),
            p=list(self.strategy_weights.values())
        )
        
        self.last_strategy = strategy_name
        strategy = self.strategies[strategy_name]
        return strategy.decide_trades(market_state)