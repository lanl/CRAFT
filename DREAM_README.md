# DREAM Algorithm for Bayesian Parameter Estimation

## What is DREAM?

**DREAM** (DiffeRential Evolution Adaptive Metropolis) is an advanced MCMC (Markov Chain Monte Carlo) algorithm designed for efficient Bayesian parameter estimation. It combines concepts from genetic algorithms (differential evolution) with traditional MCMC methods to provide robust sampling from complex posterior distributions.

### Key Advantages over Standard MCMC Methods

1. **Multiple Parallel Chains**: Runs several chains simultaneously, improving exploration
2. **Automatic Tuning**: Self-adapts proposal distribution using information from all chains
3. **Efficient for Complex Distributions**: Handles correlated parameters and multimodal distributions
4. **Outlier Detection**: Automatically detects and corrects stuck chains
5. **Subspace Sampling**: Can update subsets of parameters for high-dimensional problems

## How DREAM Works

### Core Mechanism

Instead of proposing new states using a fixed covariance matrix (like standard Metropolis-Hastings), DREAM generates proposals using **differential evolution**:

```
New Proposal = Current Position + γ × (Chain_A - Chain_B) + ε
```

Where:
- **γ** (gamma): Jump rate, typically 2.38/√(2·δ·d) where d is dimensionality
- **Chain_A, Chain_B**: Randomly selected other chains
- **ε** (epsilon): Small random noise for ergodicity

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_chains` | Number of parallel chains | 3-10 |
| `delta` | Number of chain pairs for DE | 2-3 |
| `c` | Noise scale factor | 0.1 |
| `c_star` | Ergodicity constant | 1e-12 |
| `n_crossover` | Crossover probability levels | 3 |
| `p_gamma_unity` | Probability γ=1 | 0.2 |

## Installation & Requirements

```bash
pip install numpy pandas scipy matplotlib
```

## Quick Start Example

```python
from DREAM_example import DREAMSampler
import numpy as np
from scipy.stats import multivariate_normal

# 1. Define your log posterior function
def log_posterior(params):
    # Example: 2D Gaussian
    mean = np.array([2.0, -1.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    return multivariate_normal.logpdf(params, mean=mean, cov=cov)

# 2. Create DREAM sampler
sampler = DREAMSampler(
    log_posterior=log_posterior,
    n_chains=5,        # Number of parallel chains
    n_params=2,        # Number of parameters
    delta=2            # Chain pairs for proposals
)

# 3. Initialize chains
bounds = [(-5, 5), (-5, 5)]  # Parameter bounds
sampler.initialize_chains(bounds=bounds)

# 4. Run sampling
samples, log_posts = sampler.run(
    n_iterations=3000,  # Total iterations
    burnin=500,         # Discard first 500 samples
    thin=2,             # Keep every 2nd sample
    verbose=True        # Print progress
)

# 5. Analyze results
print(f"Posterior mean: {samples.mean(axis=0)}")
print(f"Posterior std: {samples.std(axis=0)}")
```

## Detailed Usage Guide

### Step 1: Define Your Model

Your log posterior function should accept a parameter vector and return the log probability:

```python
def log_posterior(params):
    """
    Calculate log posterior probability
    
    Parameters:
    -----------
    params : np.ndarray
        Parameter vector of shape (n_params,)
        
    Returns:
    --------
    log_prob : float
        Log posterior probability
    """
    # Prior
    log_prior = -0.5 * np.sum(params**2)  # Gaussian prior
    
    # Likelihood (example: comparing model to data)
    model_predictions = my_model(params)
    log_likelihood = -0.5 * np.sum((data - model_predictions)**2 / sigma**2)
    
    return log_prior + log_likelihood
```

### Step 2: Configure DREAM Sampler

```python
sampler = DREAMSampler(
    log_posterior=log_posterior,
    n_chains=5,              # More chains = better exploration
    n_params=10,             # Your parameter dimension
    delta=2,                 # Use 2-3 chain pairs
    c=0.1,                   # Standard noise level
    n_crossover=3,           # 3 crossover levels is typical
    p_gamma_unity=0.2        # 20% chance of large jumps
)
```

**Tuning Tips:**
- **More chains** → Better for multimodal distributions, slower
- **Higher delta** → More aggressive proposals, may reduce acceptance
- **Higher c** → More exploration, lower acceptance rate
- **Higher p_gamma_unity** → More large jumps, helps escape local modes

### Step 3: Initialize Chains

**Option A: Random within bounds**
```python
bounds = [(0, 10), (-5, 5), (0, 1)]  # For 3 parameters
sampler.initialize_chains(bounds=bounds)
```

**Option B: Custom initial positions**
```python
# Start chains near a good initial guess
initial_guess = np.array([5.0, 0.0, 0.5])
noise = 0.1
initial_positions = initial_guess + np.random.randn(5, 3) * noise
sampler.initialize_chains(initial_positions=initial_positions)
```

### Step 4: Run Sampling

```python
samples, log_posts = sampler.run(
    n_iterations=10000,     # Total iterations
    burnin=2000,            # Throw away first 2000 (adaptation period)
    thin=5,                 # Keep every 5th sample (reduces correlation)
    verbose=True            # Print progress
)

# samples shape: (n_samples, n_params)
# log_posts shape: (n_samples,)
```

**Tuning Tips:**
- **Burnin**: Should be long enough for chains to converge (check trace plots)
- **Thin**: Reduces autocorrelation but gives fewer samples
- **n_iterations**: More is better, but diminishing returns after convergence

### Step 5: Diagnostics & Analysis

```python
import matplotlib.pyplot as plt

# 1. Trace plots (check convergence)
fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
for i in range(n_params):
    axes[i].plot(samples[:, i])
    axes[i].set_ylabel(f'Parameter {i+1}')
    axes[i].set_xlabel('Sample')
plt.tight_layout()
plt.show()

# 2. Posterior distributions
fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 3))
for i in range(n_params):
    axes[i].hist(samples[:, i], bins=50, density=True)
    axes[i].set_xlabel(f'Parameter {i+1}')
    axes[i].set_ylabel('Density')
plt.tight_layout()
plt.show()

# 3. Parameter correlations
import pandas as pd
df = pd.DataFrame(samples, columns=[f'param_{i+1}' for i in range(n_params)])
print(df.corr())

# 4. Summary statistics
print("Posterior mean:", samples.mean(axis=0))
print("Posterior std:", samples.std(axis=0))
print("95% credible intervals:")
for i in range(n_params):
    lower = np.percentile(samples[:, i], 2.5)
    upper = np.percentile(samples[:, i], 97.5)
    print(f"  Parameter {i+1}: [{lower:.4f}, {upper:.4f}]")
```

## Comparison with Your Current Method (Adaptive MCMC)

Your current implementation in `3_Bayes_Fit.py` uses **Adaptive Metropolis**, which is a single-chain method. Here's how DREAM compares:

| Feature | Adaptive MCMC (Current) | DREAM |
|---------|------------------------|-------|
| **Number of chains** | 1 | Multiple (3-10) |
| **Adaptation method** | Covariance matrix | Differential evolution |
| **Multimodal handling** | Poor | Good |
| **Parallel efficiency** | N/A | Excellent |
| **Convergence speed** | Moderate | Fast |
| **Parameter correlations** | Adapts slowly | Handles well |
| **Outlier detection** | No | Yes |

### When to Use DREAM vs. Adaptive MCMC

**Use DREAM when:**
- You have complex, multimodal posteriors
- Parameters are highly correlated
- You have computational resources for parallel chains
- You want robust convergence diagnostics

**Use Adaptive MCMC when:**
- Posterior is unimodal and well-behaved
- You need a simple, lightweight implementation
- Computational resources are limited
- You're already getting good results

## Running the Examples

The provided `DREAM_example.py` includes two complete examples:

### Example 1: 2D Correlated Gaussian
```bash
python DREAM_example.py
```

This will:
- Sample from a 2D Gaussian with correlation
- Generate diagnostic plots
- Save results to `DREAM_gaussian_samples.csv`
- Create visualization: `DREAM_example_results.png`

### Example 2: Banana Distribution
A challenging test case with a curved, banana-shaped distribution to demonstrate DREAM's ability to handle difficult geometries.

## Integration with Your CRAFT Workflow

To integrate DREAM into your existing `3_Bayes_Fit.py`:

```python
# 1. Import the DREAM sampler
from DREAM_example import DREAMSampler

# 2. Modify your log_prob function to accept array input
def log_prob_dream(params):
    # Convert params array to your format
    x = scaler_pars.transform(params.reshape(1, -1)).flatten()
    return log_prob(regrli, scarleri, x)

# 3. Replace AdaptiveMCMC with DREAM
sampler = DREAMSampler(
    log_posterior=log_prob_dream,
    n_chains=5,
    n_params=len(Varset2),
    delta=2
)

# 4. Initialize from your samples
initial_positions = samples_sub.sample(5).values
sampler.initialize_chains(initial_positions=initial_positions)

# 5. Run DREAM
samples_dream, log_posts_dream = sampler.run(
    n_iterations=steps,
    burnin=burnin,
    thin=1,
    verbose=True
)

# 6. Transform back and save
Fixedlog = pd.DataFrame(scaler_pars.inverse_transform(samples_dream))
Fixedlog.to_csv(output_file)
```

## Advanced Features

### Custom Proposal Distribution
You can modify the proposal generation by subclassing:

```python
class CustomDREAM(DREAMSampler):
    def generate_proposal(self, chain_idx, all_chains):
        # Your custom proposal logic
        proposal = super().generate_proposal(chain_idx, all_chains)
        # Add custom modifications
        return proposal
```

### Parallel Tempering Extension
For very challenging posteriors, consider adding parallel tempering:

```python
# Use different temperatures for different chains
temperatures = np.linspace(1.0, 2.0, n_chains)

def log_posterior_tempered(params, temp):
    return log_posterior(params) / temp
```

## References

1. **Vrugt, J. A., ter Braak, C. J. F., Diks, C. G. H., Robinson, B. A., Hyman, J. M., & Higdon, D. (2009).** 
   "Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling." 
   *International Journal of Nonlinear Sciences and Numerical Simulation*, 10(3), 273-290.

2. **ter Braak, C. J. F. (2006).** 
   "A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces." 
   *Statistics and Computing*, 16(3), 239-249.

3. **Vrugt, J. A. (2016).** 
   "Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and MATLAB implementation." 
   *Environmental Modelling & Software*, 75, 273-316.

## Troubleshooting

### Low Acceptance Rate (<10%)
- Reduce `c` parameter (less noise)
- Increase `n_chains` (more information for proposals)
- Check if posterior is well-defined

### Chains Not Converging
- Increase `burnin` period
- Check trace plots for trends
- Verify log_posterior function is correct
- Try different initial positions

### Memory Issues
- Reduce `n_chains`
- Increase `thin` to store fewer samples
- Process results in batches

### Slow Sampling
- Optimize your log_posterior function
- Use vectorization where possible
- Consider reducing `n_chains`

## Support

For questions or issues with the DREAM implementation, please refer to:
- The example code in `DREAM_example.py`
- Original DREAM papers (see References)
- CRAFT repository: https://github.com/lanl/CRAFT

---

**Created for CRAFT: Climate Risk Analysis Framework & Toolset**
