import numpy as np
import pandas as pd

def generate_corporate_portfolio(num_loans: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic corporate loan book using log-normal exposures 
    and Beta-distributed Loss Given Default (LGD).
    """
    np.random.seed(seed)
    
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC/C']
    rating_probs = [0.03, 0.07, 0.25, 0.40, 0.15, 0.08, 0.02]
    assigned_ratings = np.random.choice(ratings, size=num_loans, p=rating_probs)
    
    eads = np.random.lognormal(mean=14.5, sigma=1.0, size=num_loans)
    eads = np.clip(eads, 250000, 50000000)
    
    lgds = np.random.beta(a=4, b=5, size=num_loans)
    
    return pd.DataFrame({
        'Loan_ID': [f'CORP_{i+10001}' for i in range(num_loans)],
        'Rating': assigned_ratings,
        'EAD': np.round(eads, -3),
        'LGD': np.round(lgds, 3)
    })