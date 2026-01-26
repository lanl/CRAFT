"""
DREAM (DiffeRential Evolution Adaptive Metropolis) Algorithm Example
A multi-chain MCMC method for Bayesian parameter estimation

This implementation demonstrates the core concepts of DREAM:
- Multiple parallel chains
- Differential evolution for proposal generation
- Adaptive crossover probability
- Outlier chain detection and correction

Reference: Vrugt et al. (2009) - "Accelerating Markov chain Monte Carlo simulation 
by differential evolution with self-adaptive randomized subspace sampling"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DREAMSampler:
    """
    DREAM (DiffeRential Evolution Adaptive Metropolis) Sampler
    """
    
    def __init__(self, 
                 log_posterior: Callable,
                 n_chains: int = 3,
                 n_params: int = 2,
                 delta: int = 3,
                 c: float = 0.1,
                 c_star: float = 1e-12,
                 n_crossover: int = 3,
                 p_gamma_unity: float = 0.2):
        """
        Initialize DREAM sampler
        
        Parameters:
        -----------
        log_posterior : Callable
            Function that returns log posterior probability
        n_chains : int
            Number of parallel chains (minimum 3, recommended: 3-10)
        n_params : int
            Number of parameters to estimate
        delta : int
            Number of chain pairs used for proposal generation
        c : float
            Scaling factor for epsilon randomization
        c_star : float
            Small constant to maintain ergodicity
        n_crossover : int
            Number of crossover probabilities to choose from
        p_gamma_unity : float
            Probability of setting gamma to 1
        """
        self.log_posterior = log_posterior
        self.n_chains = max(n_chains, 3)  # Minimum 3 chains
        self.n_params = n_params
        self.delta = min(delta, n_chains // 2)
        self.c = c
        self.c_star = c_star
        self.n_crossover = n_crossover
        self.p_gamma_unity = p_gamma_unity
        
        # Storage for chains
        self.chains = None
        self.log_posteriors = None
        self.acceptance_rate = []
        
    def initialize_chains(self, initial_positions: np.ndarray = None, 
                         bounds: List[Tuple[float, float]] = None):
        """
        Initialize chain positions
        
        Parameters:
        -----------
        initial_positions : np.ndarray, optional
            Initial positions for chains (n_chains x n_params)
        bounds : List[Tuple[float, float]], optional
            Parameter bounds for random initialization
        """
        if initial_positions is not None:
            if initial_positions.shape != (self.n_chains, self.n_params):
                raise ValueError(f"Initial positions must be shape ({self.n_chains}, {self.n_params})")
            self.chains = initial_positions.copy()
        else:
            # Random initialization within bounds
            if bounds is None:
                bounds = [(-5, 5)] * self.n_params
            
            self.chains = np.zeros((self.n_chains, self.n_params))
            for i in range(self.n_params):
                self.chains[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], self.n_chains)
        
        # Calculate initial log posteriors
        self.log_posteriors = np.array([self.log_posterior(chain) for chain in self.chains])
        
    def generate_crossover_probabilities(self) -> np.ndarray:
        """Generate crossover probability vector for parameter subspace sampling"""
        # Sample from predefined crossover values
        crossover_values = np.linspace(1/self.n_crossover, 1.0, self.n_crossover)
        CR = np.random.choice(crossover_values)
        
        # Create crossover mask
        z = np.random.uniform(0, 1, self.n_params)
        return (z <= CR).astype(float)
        
    def generate_proposal(self, chain_idx: int, all_chains: np.ndarray) -> np.ndarray:
        """
        Generate proposal using differential evolution
        
        Parameters:
        -----------
        chain_idx : int
            Index of current chain
        all_chains : np.ndarray
            Current positions of all chains
            
        Returns:
        --------
        proposal : np.ndarray
            Proposed new position
        """
        # Select delta pairs of chains (excluding current chain)
        available_chains = list(range(self.n_chains))
        available_chains.remove(chain_idx)
        
        # Sample 2*delta chains for delta pairs
        selected = np.random.choice(available_chains, size=2*self.delta, replace=False)
        chain_pairs = selected.reshape(self.delta, 2)
        
        # Calculate gamma (jump rate)
        if np.random.uniform() < self.p_gamma_unity:
            gamma = 1.0
        else:
            gamma = 2.38 / np.sqrt(2 * self.delta * self.n_params)
        
        # Generate crossover probabilities
        CR_mask = self.generate_crossover_probabilities()
        
        # Calculate differential evolution step
        de_step = np.zeros(self.n_params)
        for pair in chain_pairs:
            de_step += all_chains[pair[0]] - all_chains[pair[1]]
        
        # Add randomization
        epsilon = self.c * np.random.normal(0, 1, self.n_params)
        
        # Generate proposal with subspace sampling
        proposal = all_chains[chain_idx].copy()
        update_mask = CR_mask > 0
        
        if np.any(update_mask):
            proposal[update_mask] += (gamma * de_step[update_mask] + 
                                     epsilon[update_mask] + 
                                     self.c_star * np.random.normal(0, 1, np.sum(update_mask)))
        
        return proposal
    
    def metropolis_acceptance(self, current_log_post: float, 
                             proposal_log_post: float) -> bool:
        """Metropolis acceptance criterion"""
        if proposal_log_post > current_log_post:
            return True
        
        acceptance_prob = np.exp(proposal_log_post - current_log_post)
        return np.random.uniform() < acceptance_prob
    
    def detect_outliers(self, window: int = 50) -> np.ndarray:
        """
        Detect outlier chains using Interquartile Range (IQR) method
        
        Parameters:
        -----------
        window : int
            Number of recent iterations to consider
            
        Returns:
        --------
        outliers : np.ndarray
            Boolean array indicating outlier chains
        """
        if len(self.acceptance_rate) < window:
            return np.zeros(self.n_chains, dtype=bool)
        
        recent_accepts = np.array(self.acceptance_rate[-window:])
        chain_accept_rates = recent_accepts.mean(axis=0)
        
        Q1 = np.percentile(chain_accept_rates, 25)
        Q3 = np.percentile(chain_accept_rates, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (chain_accept_rates < lower_bound) | (chain_accept_rates > upper_bound)
        return outliers
    
    def run(self, n_iterations: int, burnin: int = 500, 
            thin: int = 1, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DREAM sampling
        
        Parameters:
        -----------
        n_iterations : int
            Total number of iterations
        burnin : int
            Number of burnin iterations to discard
        thin : int
            Thinning interval for storing samples
        verbose : bool
            Print progress information
            
        Returns:
        --------
        samples : np.ndarray
            Posterior samples (n_samples x n_params)
        log_posts : np.ndarray
            Log posterior values for samples
        """
        if self.chains is None:
            raise ValueError("Chains not initialized. Call initialize_chains() first.")
        
        # Storage for all iterations
        all_chains_history = np.zeros((n_iterations, self.n_chains, self.n_params))
        all_log_posts_history = np.zeros((n_iterations, self.n_chains))
        
        # Progress tracking
        accepted_per_iteration = np.zeros((n_iterations, self.n_chains))
        
        for iteration in range(n_iterations):
            accepts = np.zeros(self.n_chains, dtype=bool)
            
            # Update each chain
            for chain_idx in range(self.n_chains):
                # Generate proposal
                proposal = self.generate_proposal(chain_idx, self.chains)
                
                # Calculate log posterior for proposal
                proposal_log_post = self.log_posterior(proposal)
                
                # Acceptance step
                if self.metropolis_acceptance(self.log_posteriors[chain_idx], 
                                              proposal_log_post):
                    self.chains[chain_idx] = proposal
                    self.log_posteriors[chain_idx] = proposal_log_post
                    accepts[chain_idx] = True
            
            # Store current state
            all_chains_history[iteration] = self.chains.copy()
            all_log_posts_history[iteration] = self.log_posteriors.copy()
            accepted_per_iteration[iteration] = accepts
            self.acceptance_rate.append(accepts)
            
            # Outlier detection and correction (after burnin)
            if iteration > burnin and iteration % 100 == 0:
                outliers = self.detect_outliers()
                if np.any(outliers):
                    # Replace outlier chains with random good chains
                    good_chains = np.where(~outliers)[0]
                    for outlier_idx in np.where(outliers)[0]:
                        replacement_idx = np.random.choice(good_chains)
                        self.chains[outlier_idx] = self.chains[replacement_idx] + \
                                                   np.random.normal(0, 0.01, self.n_params)
                        self.log_posteriors[outlier_idx] = self.log_posterior(self.chains[outlier_idx])
            
            # Progress reporting
            if verbose and (iteration + 1) % 500 == 0:
                accept_rate = np.mean(accepted_per_iteration[max(0, iteration-499):iteration+1])
                print(f"Iteration {iteration + 1}/{n_iterations} - "
                      f"Acceptance rate: {accept_rate:.3f} - "
                      f"Mean log-posterior: {np.mean(self.log_posteriors):.2f}")
        
        # Extract post-burnin samples with thinning
        samples = []
        log_posts = []
        
        for iteration in range(burnin, n_iterations, thin):
            for chain_idx in range(self.n_chains):
                samples.append(all_chains_history[iteration, chain_idx])
                log_posts.append(all_log_posts_history[iteration, chain_idx])
        
        samples = np.array(samples)
        log_posts = np.array(log_posts)
        
        if verbose:
            final_accept_rate = np.mean(accepted_per_iteration)
            print(f"\nSampling complete!")
            print(f"Overall acceptance rate: {final_accept_rate:.3f}")
            print(f"Total samples collected: {len(samples)}")
        
        return samples, log_posts
    
    def get_chain_history(self) -> np.ndarray:
        """Return full chain history for diagnostics"""
        return np.array(self.acceptance_rate)


# ============================================================================
# EXAMPLE APPLICATION: 2D Gaussian Target Distribution
# ============================================================================

def example_2d_gaussian():
    """
    Example: Sample from a 2D Gaussian distribution with correlation
    """
    print("=" * 70)
    print("DREAM Algorithm Example: 2D Correlated Gaussian")
    print("=" * 70)
    
    # True parameters
    true_mean = np.array([2.0, -1.0])
    true_cov = np.array([[1.0, 0.8],
                         [0.8, 1.5]])
    
    # Define log posterior (negative for minimization)
    def log_posterior(x):
        return multivariate_normal.logpdf(x, mean=true_mean, cov=true_cov)
    
    # Initialize DREAM sampler
    n_chains = 5
    n_params = 2
    sampler = DREAMSampler(
        log_posterior=log_posterior,
        n_chains=n_chains,
        n_params=n_params,
        delta=2,
        c=0.1,
        n_crossover=3
    )
    
    # Initialize chains with random positions
    bounds = [(-5, 5), (-5, 5)]
    sampler.initialize_chains(bounds=bounds)
    
    # Run DREAM
    print("\nRunning DREAM sampler...")
    samples, log_posts = sampler.run(
        n_iterations=3000,
        burnin=500,
        thin=2,
        verbose=True
    )
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTrue mean: {true_mean}")
    print(f"Estimated mean: {samples.mean(axis=0)}")
    print(f"\nTrue covariance:\n{true_cov}")
    print(f"\nEstimated covariance:\n{np.cov(samples.T)}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Trace plots
    axes[0, 0].plot(samples[:, 0], alpha=0.5)
    axes[0, 0].axhline(y=true_mean[0], color='r', linestyle='--', label='True value')
    axes[0, 0].set_ylabel('Parameter 1')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].legend()
    axes[0, 0].set_title('Trace Plot: Parameter 1')
    
    axes[0, 1].plot(samples[:, 1], alpha=0.5)
    axes[0, 1].axhline(y=true_mean[1], color='r', linestyle='--', label='True value')
    axes[0, 1].set_ylabel('Parameter 2')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].legend()
    axes[0, 1].set_title('Trace Plot: Parameter 2')
    
    # 2. 2D scatter plot with contours
    from scipy.stats import gaussian_kde
    
    x_grid = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
    y_grid = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # True distribution contours
    pos = np.dstack((X, Y))
    rv = multivariate_normal(true_mean, true_cov)
    Z_true = rv.pdf(pos)
    
    axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label='DREAM samples')
    axes[1, 0].contour(X, Y, Z_true, colors='red', alpha=0.6, levels=5)
    axes[1, 0].plot(true_mean[0], true_mean[1], 'r*', markersize=15, label='True mean')
    axes[1, 0].set_xlabel('Parameter 1')
    axes[1, 0].set_ylabel('Parameter 2')
    axes[1, 0].legend()
    axes[1, 0].set_title('2D Posterior Distribution')
    
    # 3. Marginal histograms
    axes[1, 1].hist(samples[:, 0], bins=50, alpha=0.6, density=True, label='Parameter 1')
    axes[1, 1].hist(samples[:, 1], bins=50, alpha=0.6, density=True, label='Parameter 2')
    axes[1, 1].axvline(x=true_mean[0], color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=true_mean[1], color='orange', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Parameter Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].set_title('Marginal Distributions')
    
    plt.tight_layout()
    plt.savefig('DREAM_example_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'DREAM_example_results.png'")
    
    return samples, log_posts, sampler


# ============================================================================
# EXAMPLE APPLICATION: Banana-shaped Distribution (challenging case)
# ============================================================================

def example_banana_distribution():
    """
    Example: Sample from a banana-shaped (Rosenbrock) distribution
    This is a challenging test case for MCMC samplers
    """
    print("\n" + "=" * 70)
    print("DREAM Algorithm Example: Banana-shaped Distribution")
    print("=" * 70)
    
    # Banana-shaped distribution (Rosenbrock)
    def log_posterior(x):
        a = 1.0
        b = 100.0
        term1 = (a - x[0])**2
        term2 = b * (x[1] - x[0]**2)**2
        return -0.5 * (term1 + term2) / 20.0  # Scale for better sampling
    
    # Initialize DREAM sampler
    sampler = DREAMSampler(
        log_posterior=log_posterior,
        n_chains=6,
        n_params=2,
        delta=2,
        c=0.05
    )
    
    # Initialize chains
    initial_positions = np.random.randn(6, 2) * 0.5 + np.array([1.0, 1.0])
    sampler.initialize_chains(initial_positions=initial_positions)
    
    # Run DREAM
    print("\nRunning DREAM sampler on banana distribution...")
    samples, log_posts = sampler.run(
        n_iterations=5000,
        burnin=1000,
        thin=3,
        verbose=True
    )
    
    # Visualize banana distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample scatter
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5)
    axes[0].set_xlabel('Parameter 1')
    axes[0].set_ylabel('Parameter 2')
    axes[0].set_title('DREAM Samples from Banana Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Trace plot
    axes[1].plot(log_posts, alpha=0.7)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Log Posterior')
    axes[1].set_title('Log Posterior Trace')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('DREAM_banana_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'DREAM_banana_results.png'")
    
    return samples, log_posts, sampler


if __name__ == "__main__":
    # Run example 1: 2D Gaussian
    samples_gauss, log_posts_gauss, sampler_gauss = example_2d_gaussian()
    
    # Run example 2: Banana distribution
    samples_banana, log_posts_banana, sampler_banana = example_banana_distribution()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    
    # Save samples to CSV
    df_gauss = pd.DataFrame(samples_gauss, columns=['param1', 'param2'])
    df_gauss['log_posterior'] = log_posts_gauss
    df_gauss.to_csv('DREAM_gaussian_samples.csv', index=False)
    print("\nGaussian samples saved to 'DREAM_gaussian_samples.csv'")
    
    df_banana = pd.DataFrame(samples_banana, columns=['param1', 'param2'])
    df_banana['log_posterior'] = log_posts_banana
    df_banana.to_csv('DREAM_banana_samples.csv', index=False)
    print("Banana samples saved to 'DREAM_banana_samples.csv'")
