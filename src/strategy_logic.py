import pandas as pd
import numpy as np

class HMMStrategy:
    def __init__(self, hmm_model, feature_cols=["log_return", "volatility"], bull_regime=1):

        self.model = hmm_model
        self.feature_cols = feature_cols
        self.bull_regime = bull_regime
    
    def _discretize_weight(self, prob):
        if prob >= 0.75:
            return 1.0
        elif prob >= 0.5:
            return 0.50
        else:
            return 0.0
    
    def generate_signals(self, df):
        X = df[self.feature_cols].values
        probs = self.model.predict_proba(X)
        bull_probs = probs[:,self.bull_regime]
        target_weights = pd.Series(bull_probs, index=df.index).shift(1).fillna(0)
        final_weights = target_weights.apply(self._discretize_weight)
        return final_weights