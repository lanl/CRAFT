# MCMC Sampler Configuration Guide

## Overview

`3_Bayes_Fit.py` now supports two MCMC sampling methods:
- **Adaptive MCMC** (original, single-chain method)
- **DREAM** (multi-chain differential evolution method)

You can switch between them by editing the `config.xml` file.

## Quick Start

### Using Adaptive MCMC (Default)

In `config.xml`, set:
```xml
<sampler_method>adaptive</sampler_method>
```

Then run:
```bash
python 3_Bayes_Fit.py
```

### Using DREAM

In `config.xml`, set:
```xml
<sampler_method>dream</sampler_method>
```

Then run:
```bash
python 3_Bayes_Fit.py
```

## Configuration Details

### config.xml Structure

```xml
<bayes_settings>
    <!-- Common settings (used by both samplers) -->
    <initialpos>45</initialpos>
    <steps>10000</steps>
    <adapt_interval>1000</adapt_interval>
    <burnin>1000</burnin>
    <model_list>['CRAFT_practice_FullFATES_GPP','CRAFT_practice_FullH2OSOIvalue1']</model_list>
    <scaler_list>['CRAFT_practice_FullFATES_GPP_Scalar','CRAFT_practice_FullH2OSOIvalue1_Scalar']</scaler_list>
    <obs_list>['MonthlyGPP','SWC10']</obs_list>
    <list1>['MonthlyGPP','SWC10']</list1>
    <samples_file>FATES_Param_samples_bci.csv</samples_file>
    <obsfile>Example_Obs_data/Syntheticg_Set2_63025.csv</obsfile>
    <output_file>data/MCMC_runs/TestMC.csv</output_file>
    
    <!-- Sampler selection: 'adaptive' or 'dream' -->
    <sampler_method>adaptive</sampler_method>
    
    <!-- DREAM-specific settings (only used when sampler_method='dream') -->
    <dream_n_chains>5</dream_n_chains>
    <dream_delta>2</dream_delta>
    <dream_c>0.1</dream_c>
    <dream_c_star>1e-12</dream_c_star>
    <dream_n_crossover>3</dream_n_crossover>
    <dream_p_gamma_unity>0.2</dream_p_gamma_unity>
    <dream_thin>1</dream_thin>
</bayes_settings>
```

### Parameter Reference

#### Common Parameters (Both Samplers)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `initialpos` | Starting position index (Adaptive only) | `45` |
| `steps` | Total number of MCMC iterations | `10000` |
| `adapt_interval` | Adaptation frequency (Adaptive only) | `1000` |
| `burnin` | Burnin period (samples to discard) | `1000` |
| `output_file` | Path to save results | `data/MCMC_runs/TestMC.csv` |
| `sampler_method` | Choose `'adaptive'` or `'dream'` | `adaptive` |

#### DREAM-Specific Parameters

| Parameter | Description | Recommended | Range |
|-----------|-------------|-------------|-------|
| `dream_n_chains` | Number of parallel chains | `5` | 3-10 |
| `dream_delta` | Number of chain pairs for proposals | `2` | 1-3 |
| `dream_c` | Noise scale factor | `0.1` | 0.01-0.2 |
| `dream_c_star` | Ergodicity constant | `1e-12` | 1e-15 to 1e-10 |
| `dream_n_crossover` | Crossover probability levels | `3` | 2-5 |
| `dream_p_gamma_unity` | Probability of large jumps | `0.2` | 0.1-0.3 |
| `dream_thin` | Thinning interval | `1` | 1-10 |

## Comparison: Adaptive MCMC vs. DREAM

### Adaptive MCMC
**Pros:**
- Simpler, fewer parameters to tune
- Faster for well-behaved posteriors
- Lower memory usage
- Good for quick tests

**Cons:**
- Single chain (harder to diagnose convergence)
- Slower adaptation to complex geometries
- May get stuck in local modes
- Less efficient for correlated parameters

**Best for:**
- Unimodal posteriors
- Quick exploratory runs
- Limited computational resources
- When you're already getting good results

### DREAM
**Pros:**
- Multiple chains improve convergence diagnostics
- Better exploration of complex posteriors
- Handles parameter correlations well
- Automatic outlier detection
- More efficient for multimodal distributions

**Cons:**
- More parameters to configure
- Higher memory usage
- Slower per iteration (but often fewer iterations needed)
- Requires understanding of additional parameters

**Best for:**
- Complex, multimodal posteriors
- Highly correlated parameters
- Production runs requiring robust results
- When convergence is difficult with Adaptive MCMC

## Tuning Guidelines

### For Adaptive MCMC

1. **Initial Testing** (Quick runs)
   ```xml
   <steps>5000</steps>
   <burnin>500</burnin>
   <adapt_interval>500</adapt_interval>
   ```

2. **Production Runs**
   ```xml
   <steps>50000</steps>
   <burnin>5000</burnin>
   <adapt_interval>1000</adapt_interval>
   ```

### For DREAM

1. **Initial Testing** (Quick runs)
   ```xml
   <sampler_method>dream</sampler_method>
   <steps>3000</steps>
   <burnin>500</burnin>
   <dream_n_chains>3</dream_n_chains>
   <dream_thin>1</dream_thin>
   ```

2. **Standard Configuration**
   ```xml
   <sampler_method>dream</sampler_method>
   <steps>10000</steps>
   <burnin>2000</burnin>
   <dream_n_chains>5</dream_n_chains>
   <dream_delta>2</dream_delta>
   <dream_thin>2</dream_thin>
   ```

3. **High-Quality Sampling** (Long runs)
   ```xml
   <sampler_method>dream</sampler_method>
   <steps>50000</steps>
   <burnin>10000</burnin>
   <dream_n_chains>8</dream_n_chains>
   <dream_delta>3</dream_delta>
   <dream_thin>5</dream_thin>
   ```

4. **Difficult Posterior** (Multimodal, highly correlated)
   ```xml
   <sampler_method>dream</sampler_method>
   <steps>100000</steps>
   <burnin>20000</burnin>
   <dream_n_chains>10</dream_n_chains>
   <dream_delta>3</dream_delta>
   <dream_c>0.15</dream_c>
   <dream_p_gamma_unity>0.3</dream_p_gamma_unity>
   <dream_thin>10</dream_thin>
   ```

## Output Format

### Adaptive MCMC Output
CSV file with parameter columns (no column names by default):
```
param1, param2, param3, ...
0.523,  -1.234,  0.891, ...
0.531,  -1.221,  0.903, ...
...
```

### DREAM Output
CSV file with parameter columns AND log posterior:
```
param1, param2, param3, ..., log_posterior
0.523,  -1.234,  0.891, ..., -45.32
0.531,  -1.221,  0.903, ..., -45.21
...
```

## Troubleshooting

### Issue: "DREAM sampler selected but DREAM_example.py not found!"

**Solution:** Make sure `DREAM_example.py` is in the same directory as `3_Bayes_Fit.py`.

### Issue: Low acceptance rate with DREAM

**Solutions:**
1. Reduce `dream_c` (less noise): `<dream_c>0.05</dream_c>`
2. Increase number of chains: `<dream_n_chains>8</dream_n_chains>`
3. Check that your log posterior function is working correctly

### Issue: Chains not converging with DREAM

**Solutions:**
1. Increase burnin: `<burnin>5000</burnin>`
2. Increase total iterations: `<steps>20000</steps>`
3. Try different initial positions
4. Check posterior function for issues (infinities, NaNs)

### Issue: DREAM is slow

**Solutions:**
1. Reduce number of chains: `<dream_n_chains>3</dream_n_chains>`
2. Increase thinning: `<dream_thin>5</dream_thin>`
3. Use Adaptive MCMC for quick tests
4. Optimize your log posterior function

## Example Workflow

### 1. Start with Adaptive MCMC for Quick Testing
```xml
<sampler_method>adaptive</sampler_method>
<steps>5000</steps>
<burnin>500</burnin>
```

Run and check if results look reasonable.

### 2. Switch to DREAM for Production
```xml
<sampler_method>dream</sampler_method>
<steps>10000</steps>
<burnin>2000</burnin>
<dream_n_chains>5</dream_n_chains>
<dream_thin>2</dream_thin>
```

### 3. Analyze Results

Check convergence:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('data/MCMC_runs/TestMC.csv')

# Plot trace plots
for col in results.columns[:-1]:  # Skip log_posterior
    plt.figure()
    plt.plot(results[col])
    plt.title(f'Trace: {col}')
    plt.ylabel('Parameter Value')
    plt.xlabel('Sample')
    plt.show()

# Check log posterior
plt.figure()
plt.plot(results['log_posterior'])
plt.title('Log Posterior Trace')
plt.show()

# Summary statistics
print(results.describe())
```

## Additional Resources

- **DREAM Algorithm Details**: See `DREAM_README.md`
- **DREAM Examples**: Run `python DREAM_example.py`
- **Original Paper**: Vrugt et al. (2009), Int. J. Nonlinear Sci. Numer. Simul.

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│           SAMPLER SELECTION                     │
├─────────────────────────────────────────────────┤
│ Adaptive: Fast, simple, single chain           │
│   <sampler_method>adaptive</sampler_method>     │
│                                                 │
│ DREAM: Robust, multi-chain, complex posteriors │
│   <sampler_method>dream</sampler_method>        │
│   <dream_n_chains>5</dream_n_chains>            │
└─────────────────────────────────────────────────┘
```

---

**Last Updated**: January 22, 2026
**CRAFT Version**: With integrated DREAM sampler
