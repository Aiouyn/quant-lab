"""
Statistical & Microstructure Metrics Engine
Includes Information Coefficient (IC), Deflated Sharpe Ratio (DSR), and MinTRL
Based on Marcos López de Prado's "Advances in Financial Machine Learning"
"""

import numpy as np
import scipy.stats as stats
import polars as pl
from typing import Dict, Any, List

def compute_ic(series_x: np.ndarray, series_y: np.ndarray) -> Dict[str, float]:
    """Computes Pearson and Spearman Information Coefficients."""
    mask = ~(np.isnan(series_x) | np.isnan(series_y))
    x = series_x[mask]
    y = series_y[mask]

    if len(x) < 3:
        return {'pearson': 0.0, 'spearman': 0.0, 'p_value': 1.0}

    pearson_corr, p_val = stats.pearsonr(x, y)
    spearman_corr, _ = stats.spearmanr(x, y)

    return {
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'p_value': float(p_val)
    }

def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    sample_len: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Calculates the Deflated Sharpe Ratio (DSR).
    Penalizes for multiple hypothesis testing (num_trials tested across the lab).
    Returns probability in [0, 1] that the observed Sharpe is genuine alpha and not luck.
    """
    if num_trials <= 1:
        # Standard unadjusted t-test probability
        z = observed_sr * np.sqrt(sample_len - 1)
        return float(stats.norm.cdf(z))

    # Expected maximum Sharpe under null hypothesis (Euler-Mascheroni approximation)
    euler = 0.57721566490153286
    sr_star = (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / num_trials) + euler * stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e))

    # Variance of the Sharpe ratio estimator
    sr_std = np.sqrt((1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)) / max(sample_len - 1, 1))

    if sr_std <= 0:
        return 0.0

    z_stat = (observed_sr - sr_star) / sr_std
    dsr = float(stats.norm.cdf(z_stat))
    return dsr

def minimum_track_record_length(
    observed_sr: float,
    target_sr: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05
) -> float:
    """Calculates Minimum Track Record Length (MinTRL) required for statistical significance."""
    z_crit = stats.norm.ppf(1.0 - alpha)
    denom = (observed_sr - target_sr) ** 2
    if denom <= 0:
        return float('inf')

    var_factor = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr ** 2)
    min_len = 1.0 + var_factor * ((z_crit) ** 2) / denom
    return float(min_len)
