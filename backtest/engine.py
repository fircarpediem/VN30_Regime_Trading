import vectorbt as vbt

class VectorBTBacktester:
    def __init__(self, init_cash=100_000_000, fees=0.0015, freq="D"):
        self.init_cash = init_cash
        self.fees = fees
        self.freq = freq
        self.portfolio = None

    def run(self, close_prices, target_weights):
        self.portfolio = vbt.Portfolio.from_orders(
            close = close_prices,
            size = target_weights,
            size_type = "targetpercent",
            init_cash = self.init_cash,
            fees = self.fees,
            freq = self.freq
        )
        return self.portfolio
    
    def performance_stats(self):
        if self.portfolio is None:
            raise ValueError("No portfolio found. Please run the backtest first.")
        print(self.portfolio.stats())
