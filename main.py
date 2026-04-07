import pandas as pd
import warnings
from hmmlearn.hmm import GaussianHMM
from src.strategy_logic import HMMStrategy
from backtest.engine import VectorBTBacktester

warnings.filterwarnings("ignore")

def main():
    #Load data
    df = pd.read_csv("data/processed/vn30_features.csv", parse_dates=True, index_col = 'time')

    #Training
    X = df[["log_return", "volatility"]].values
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=10000, random_state=42)
    hmm_model.fit(X)

    strategy = HMMStrategy(hmm_model, feature_cols=["log_return", "volatility"], bull_regime=1)
    target_weights = strategy.generate_signals(df)
    backtester = VectorBTBacktester(init_cash=100_000_000, fees=0.0015, freq="D")
    portfolio = backtester.run(df["close"], target_weights)

    backtester.performance_stats()
if __name__ == "__main__":
    main()