import numpy as np
import pandas as pd
from scipy.linalg import logm, expm
from portfolio_generator import generate_corporate_portfolio
class CreditPortfolio:
    """
    An object-oriented model for assessing corporate credit risk using 
    continuous-time Markov chains and Monte Carlo simulations.
    """
    
    def __init__(self, raw_transition_matrix: pd.DataFrame, loan_book: pd.DataFrame):
        self.raw_matrix = raw_transition_matrix
        self.portfolio = loan_book
        
        # Initialize core matrices upon instantiation
        self.adjusted_matrix = self._normalize_matrix(self.raw_matrix)
        self.generator_matrix = self._compute_generator_matrix(self.adjusted_matrix)

    def _normalize_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Removes unrated (NR) entities and normalizes row sums to 1."""
        active_states = matrix.drop(columns=['NR'], errors='ignore')
        return active_states.div(active_states.sum(axis=1), axis=0)

    def _compute_generator_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Computes the continuous-time generator matrix (Q) using matrix logarithms."""
        Q_raw = logm(matrix.values)
        Q_clean = np.copy(Q_raw)
        
        # Clean mathematical artifacts: off-diagonals must be non-negative
        for i in range(Q_clean.shape[0]):
            for j in range(Q_clean.shape[1]):
                if i != j and Q_clean[i, j] < 0:
                    Q_clean[i, j] = 0.0
                    
        # Diagonal elements must equal the negative sum of off-diagonals
        for i in range(Q_clean.shape[0]):
            Q_clean[i, i] = 0.0
            Q_clean[i, i] = -np.sum(Q_clean[i, :])
            
        return pd.DataFrame(Q_clean, index=matrix.index, columns=matrix.columns)

    def get_transition_matrix(self, years: float) -> pd.DataFrame:
        """Calculates the probability transition matrix for any continuous time horizon."""
        P_t = expm(self.generator_matrix.values * years)
        return pd.DataFrame(P_t, index=self.adjusted_matrix.index, columns=self.adjusted_matrix.columns)

    def calculate_expected_loss(self, years: float) -> pd.DataFrame:
        """Calculates Expected Credit Loss (ECL) for the portfolio over a given horizon."""
        P_t_df = self.get_transition_matrix(years)
        
        # Assuming 'D' is the default state column
        pd_t = P_t_df['D'] 
        
        # Map Probability of Default (PD) to the current portfolio ratings
        results = self.portfolio.copy()
        results[f'PD_{years}Y'] = results['Rating'].map(pd_t)
        
        # ECL = PD * LGD * EAD
        results['ECL'] = results[f'PD_{years}Y'] * results['LGD'] * results['EAD']
        return results

    def calculate_value_at_risk(self, years: float, n_simulations: int = 10000, confidence_level: float = 99.0) -> dict:
        """
        Runs a Monte Carlo simulation to calculate Value at Risk (VaR) and Unexpected Loss.
        """
        P_t_df = self.get_transition_matrix(years)
        cdf_matrix = P_t_df.cumsum(axis=1)
        
        initial_ratings = self.portfolio['Rating'].values
        eads = self.portfolio['EAD'].values
        lgds = self.portfolio['LGD'].values
        
        # Fetch the CDF arrays for the specific initial ratings of our portfolio
        loan_cdfs = cdf_matrix.loc[initial_ratings].values
        
        simulated_losses = np.zeros(n_simulations)
        random_draws = np.random.uniform(0, 1, size=(n_simulations, len(self.portfolio)))
        
        # Default state index (assuming 'D' is the last state, index 7 in an 8-state system)
        default_index = len(self.adjusted_matrix.columns) - 1 

        for i in range(n_simulations):
            draws = random_draws[i]
            # Find where the random draw lands in the CDF to determine the final rating state
            final_states = (loan_cdfs > draws[:, None]).argmax(axis=1)
            
            # Filter for loans that hit the default state
            defaults = (final_states == default_index)
            simulated_losses[i] = np.sum(eads[defaults] * lgds[defaults])
            
        var_threshold = np.percentile(simulated_losses, confidence_level)
        ecl_df = self.calculate_expected_loss(years)
        total_ecl = ecl_df['ECL'].sum()
        
        return {
            "Total_EAD": np.sum(eads),
            "Expected_Credit_Loss": total_ecl,
            f"VaR_{confidence_level}": var_threshold,
            "Unexpected_Loss": var_threshold - total_ecl
        }


raw_matrix = pd.read_csv('Transition_rates.csv', index_col=0)
portfolio_data = pd.DataFrame({
    'Loan_ID': ['L101', 'L102', 'L103', 'L104', 'L105'],
    'Rating': ['AAA', 'A', 'BBB', 'BB', 'CCC/C'],
    'EAD': [15000000, 8000000, 12000000, 4500000, 1000000],
    'LGD': [0.40, 0.40, 0.45, 0.50, 0.65]
})

if __name__ == "__main__":
    # 1. Load the real-world transition matrix
    raw_matrix = pd.read_csv('Transition_rates.csv', index_col=0)
    
    # 2. Generate the large synthetic portfolio dynamically
    print("Generating 100,000-loan corporate portfolio...")
    large_portfolio = generate_corporate_portfolio(num_loans=100000)
    
    # 3. Run the simulation
    print("Running 10,000-path Monte Carlo simulations...")
    risk_model = CreditPortfolio(raw_matrix, large_portfolio)
    risk_metrics = risk_model.calculate_value_at_risk(years=5, n_simulations=10000)
    
    # 4. Print formatted results
    print("\n--- Portfolio Risk Metrics ---")
    print(f"Total EAD:            ${risk_metrics['Total_EAD']:,.2f}")
    print(f"Expected Credit Loss: ${risk_metrics['Expected_Credit_Loss']:,.2f}")
    print(f"99% Value at Risk:    ${risk_metrics['VaR_99.0']:,.2f}")
    print(f"Unexpected Loss:      ${risk_metrics['Unexpected_Loss']:,.2f}")