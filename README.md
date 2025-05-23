# Agent-Based Financial Market Simulation
## Stochastic vs Deterministic Trading Strategies

This project implements an agent-based simulation of a financial market to study the evolutionary dynamics between deterministic and stochastic trading strategies.

## Project Structure

```
├── market.py              # Market mechanics and price formation
├── traders.py             # Trader classes and strategies
├── simulation.py          # Simulation engine and analysis
├── run_experiments.py     # Main experiment runner
├── results/              # Output directory for results
└── README.md             # This file
```

## Components

### 1. Market Environment (`market.py`)

The market simulation includes:
- **4 Stocks** with different characteristics:
  - Stock A: High volatility, trending
  - Stock B: Mean-reverting, low volatility
  - Stock C: Random walk
  - Stock D: Cyclical patterns

- **Price Formation**: Prices are determined by:
  - Supply/demand imbalance from trader orders
  - Random noise component
  - Stock-specific characteristics (trend, mean reversion, cycles)
  - Price impact coefficient

### 2. Trading Strategies (`traders.py`)

#### Deterministic Strategies
1. **Momentum Trader**: Buys rising stocks, sells falling ones
   - Uses lookback period to calculate returns
   - Trades when momentum exceeds threshold

2. **Mean Reversion Trader**: Assumes prices revert to historical mean
   - Calculates z-scores of current prices
   - Buys when oversold, sells when overbought

3. **Value Trader**: Trades based on fundamental value estimates
   - Maintains internal estimates of stock values
   - Buys undervalued, sells overvalued stocks

#### Stochastic Strategies
1. **Fixed Stochastic Trader**: Uses fixed probability matrix
   - Randomly selects from deterministic strategies
   - Probabilities remain constant throughout simulation

2. **Adaptive Stochastic Trader**: Dynamically adjusts strategy weights
   - Tracks performance of each strategy
   - Updates probabilities based on recent returns
   - Uses softmax transformation for weight updates

### 3. Simulation Engine (`simulation.py`)

- **MarketSimulation**: Core simulation class
  - Manages traders and market interactions
  - Tracks performance metrics
  - Checks for convergence to equilibrium
  
- **ExperimentRunner**: Manages multiple experiments
  - Tests different trader distributions
  - Saves results and generates plots
  - Analyzes equilibrium patterns

### 4. Key Features

- **Portfolio Management**: Each trader maintains cash and stock holdings
- **Risk Management**: Position sizing based on confidence and risk tolerance
- **Performance Tracking**: Returns, Sharpe ratios, strategy evolution
- **Convergence Detection**: Monitors strategy proportions for stability
- **Visualization**: Plots for strategy evolution, prices, and performance

## Running the Experiments

### Requirements
```bash
pip install numpy pandas matplotlib seaborn
```

### Basic Usage
```python
python run_experiments.py
```

This will:
1. Run multiple experiments with different trader distributions
2. Generate plots for each experiment
3. Save detailed results to JSON files
4. Create a summary CSV with key metrics
5. Print equilibrium analysis

### Custom Experiments
```python
from simulation import MarketSimulation

# Define custom configuration
config = {
    'initial_cash': 10000,
    'risk_tolerance': 0.5
}

# Define trader distribution
distribution = {
    'momentum': 20,
    'mean_reversion': 20,
    'value': 20,
    'fixed_stochastic': 20,
    'adaptive_stochastic': 20
}

# Run simulation
sim = MarketSimulation(config)
results = sim.run_simulation(distribution, max_periods=1000)

# Plot results
sim.plot_results(results, 'my_results.png')
```

## Key Assumptions

1. **Market Structure**:
   - Perfect liquidity (can always trade at market price)
   - No transaction costs
   - Synchronous trading (all agents trade simultaneously)
   - Public information (all see same price history)

2. **Trader Behavior**:
   - Traders are price takers (no market manipulation)
   - Fixed total capital in the system
   - No leverage or short selling
   - Risk tolerance affects position sizing

3. **Price Formation**:
   - Prices respond to supply/demand imbalance
   - Each stock has inherent characteristics
   - Random noise represents external factors

## Output Analysis

The simulation produces several outputs:

1. **Convergence Analysis**: Whether strategy proportions stabilize
2. **Performance Metrics**: Average returns and Sharpe ratios by strategy
3. **Evolution Plots**: How strategies and prices evolve over time
4. **Distribution Analysis**: Final equilibrium proportions

## Extending the Project

### Adding New Strategies
1. Create a new class inheriting from `Trader` in `traders.py`
2. Implement the `decide_trades()` method
3. Add to the initialization in `simulation.py`

### Modifying Market Dynamics
1. Adjust stock characteristics in `_initialize_market()`
2. Modify price formation in `calculate_market_clearing_price()`
3. Add new market features (e.g., dividends, splits)

### Different Experiments
1. Modify distributions in `create_distributions()`
2. Adjust convergence criteria
3. Add new performance metrics

## Research Questions Addressed

1. Do deterministic or stochastic strategies dominate in equilibrium?
2. How does initial distribution affect final equilibrium?
3. Can adaptive strategies consistently outperform fixed strategies?
4. How does market structure (stock characteristics) affect strategy success?
5. What role does strategy diversity play in market stability?

## Future Enhancements

- Add transaction costs and market frictions
- Implement more sophisticated learning algorithms
- Include fundamental news events
- Add network effects between traders
- Implement short selling and leverage
- Include more complex order types (limit orders, stop losses)