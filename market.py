import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class Stock:
    """Represents a tradable stock in the market"""
    symbol: str
    current_price: float
    fundamental_value: float
    volatility: float
    drift: float
    price_impact: float
    characteristic: str  # 'trending', 'mean_reverting', 'random_walk', 'cyclical'
    
    def update_fundamental(self):
        """Update fundamental value based on stock characteristics"""
        if self.characteristic == 'trending':
            self.fundamental_value *= (1 + self.drift)
        elif self.characteristic == 'cyclical':
            # Simple sinusoidal pattern
            time_factor = np.sin(np.random.random() * 2 * np.pi)
            self.fundamental_value *= (1 + 0.02 * time_factor)
        # mean_reverting and random_walk keep constant fundamental

class Market:
    """Simulates a financial market with multiple stocks and traders"""
    
    def __init__(self, stocks: List[Stock], initial_cash: float = 10000):
        self.stocks = {stock.symbol: stock for stock in stocks}
        self.initial_cash = initial_cash
        self.order_book = {symbol: {'buy': [], 'sell': []} for symbol in self.stocks}
        self.price_history = {symbol: [stock.current_price] for symbol, stock in self.stocks.items()}
        self.volume_history = {symbol: [] for symbol in self.stocks}
        self.time_step = 0
        
    def clear_order_book(self):
        """Reset order book for new trading period"""
        self.order_book = {symbol: {'buy': [], 'sell': []} for symbol in self.stocks}
    
    def submit_order(self, trader_id: int, symbol: str, order_type: str, quantity: int, price: float = None):
        """Submit a buy or sell order to the order book"""
        if symbol not in self.stocks:
            return
        
        order = {
            'trader_id': trader_id,
            'quantity': quantity,
            'price': price if price else self.stocks[symbol].current_price
        }
        
        if order_type == 'buy':
            self.order_book[symbol]['buy'].append(order)
        elif order_type == 'sell':
            self.order_book[symbol]['sell'].append(order)
    
    def calculate_market_clearing_price(self, symbol: str) -> Tuple[float, int]:
        """Calculate the market clearing price based on supply and demand"""
        stock = self.stocks[symbol]
        buy_orders = self.order_book[symbol]['buy']
        sell_orders = self.order_book[symbol]['sell']
        
        # Calculate total demand and supply
        total_demand = sum(order['quantity'] for order in buy_orders)
        total_supply = sum(order['quantity'] for order in sell_orders)
        
        # Always update price even with no orders to maintain market dynamics
        # Add noise component
        noise = np.random.normal(0, stock.volatility)
        
        # Calculate supply/demand imbalance
        if total_demand + total_supply > 0:
            imbalance = (total_demand - total_supply) / (total_demand + total_supply)
            # Price impact from order imbalance
            price_impact = stock.price_impact * imbalance
            volume = min(total_demand, total_supply)
        else:
            # No orders, but still add some market movement
            price_impact = 0
            volume = 0
        
        # Mean reversion factor for mean-reverting stocks
        mean_reversion = 0
        if stock.characteristic == 'mean_reverting':
            mean_reversion = -0.05 * (stock.current_price / stock.fundamental_value - 1)
        
        # Calculate new price
        price_change = stock.drift + price_impact + noise + mean_reversion
        new_price = stock.current_price * (1 + price_change)
        
        # Ensure price stays positive
        new_price = max(new_price, 0.01)
        
        return new_price, volume
    
    def execute_trades(self) -> Dict[str, List[Dict]]:
        """Execute all trades in the order book and return executed trades"""
        executed_trades = {}
        
        for symbol in self.stocks:
            new_price, volume = self.calculate_market_clearing_price(symbol)
            
            # Update stock price
            self.stocks[symbol].current_price = new_price
            self.stocks[symbol].update_fundamental()
            
            # Record price and volume
            self.price_history[symbol].append(new_price)
            self.volume_history[symbol].append(volume)
            
            # Match orders (simplified - all orders at market price)
            buy_orders = sorted(self.order_book[symbol]['buy'], 
                              key=lambda x: x['price'], reverse=True)
            sell_orders = sorted(self.order_book[symbol]['sell'], 
                               key=lambda x: x['price'])
            
            trades = []
            buy_idx = sell_idx = 0
            remaining_volume = volume
            
            while (buy_idx < len(buy_orders) and sell_idx < len(sell_orders) 
                   and remaining_volume > 0):
                buy_order = buy_orders[buy_idx]
                sell_order = sell_orders[sell_idx]
                
                trade_quantity = min(buy_order['quantity'], sell_order['quantity'], 
                                   remaining_volume)
                
                if trade_quantity > 0:
                    trades.append({
                        'buyer_id': buy_order['trader_id'],
                        'seller_id': sell_order['trader_id'],
                        'quantity': trade_quantity,
                        'price': new_price
                    })
                    
                    buy_order['quantity'] -= trade_quantity
                    sell_order['quantity'] -= trade_quantity
                    remaining_volume -= trade_quantity
                
                if buy_order['quantity'] == 0:
                    buy_idx += 1
                if sell_order['quantity'] == 0:
                    sell_idx += 1
            
            executed_trades[symbol] = trades
        
        self.time_step += 1
        self.clear_order_book()
        
        return executed_trades
    
    def get_price_history(self, symbol: str, lookback: int = None) -> np.ndarray:
        """Get price history for a stock"""
        if symbol not in self.stocks:
            return np.array([])
        
        history = self.price_history[symbol]
        if lookback and len(history) > lookback:
            return np.array(history[-lookback:])
        return np.array(history)
    
    def get_returns(self, symbol: str, periods: int = 1) -> float:
        """Calculate returns over specified periods"""
        history = self.get_price_history(symbol)
        if len(history) < periods + 1:
            return 0.0
        
        return (history[-1] / history[-periods-1]) - 1
    
    def get_volatility(self, symbol: str, lookback: int = 20) -> float:
        """Calculate historical volatility"""
        history = self.get_price_history(symbol, lookback + 1)
        if len(history) < 2:
            return 0.0
        
        returns = np.diff(history) / history[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def get_market_state(self) -> Dict:
        """Get current market state for all stocks"""
        state = {}
        for symbol, stock in self.stocks.items():
            state[symbol] = {
                'price': stock.current_price,
                'returns_1': self.get_returns(symbol, 1),
                'returns_5': self.get_returns(symbol, 5),
                'returns_20': self.get_returns(symbol, 20),
                'volatility': self.get_volatility(symbol),
                'volume': self.volume_history[symbol][-1] if self.volume_history[symbol] else 0
            }
        return state