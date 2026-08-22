"""
Lag-correlation utilities for monetary / inflation / labor-market analysis.
Pure pandas / numpy / matplotlib – no extra dependencies.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def lag_corr_stats(x: pd.Series, y: pd.Series, max_lag: int = 21) -> pd.DataFrame:
    """
    For each lag k = 0 … max_lag compute:
      - Pearson correlation
      - two-sided p-value (t-test, normal approximation)
      - approximate 95 % CI for the correlation (Fisher z-transform)
      - OLS slope (x → y)
      - number of observations
    """
    rows = []
    for lag in range(max_lag + 1):
        paired = pd.concat([x, y.shift(-lag)], axis=1).dropna()
        n = len(paired)
        if n < 10:
            continue
        xx = paired.iloc[:, 0].values
        yy = paired.iloc[:, 1].values

        r = np.corrcoef(xx, yy)[0, 1]

        t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))

        z = 0.5 * np.log((1 + r) / (1 - r + 1e-12))
        se_z = 1 / np.sqrt(n - 3)
        ci_low  = np.tanh(z - 1.96 * se_z)
        ci_high = np.tanh(z + 1.96 * se_z)

        slope = np.polyfit(xx, yy, 1)[0]

        rows.append({
            'lag': lag,
            'corr': r,
            'p_value': p,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'slope': slope,
            'n_obs': n
        })
    return pd.DataFrame(rows).set_index('lag')


def plot_lag_correlation(x: pd.Series, y: pd.Series, max_lag: int = 21,
                         figsize: tuple = (12, 6), show_table: bool = True,
                         title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Compute lag correlations, optionally show a styled table,
    and plot the coefficients with Fisher-z confidence bands.

    Parameters
    ----------
    x, y : pd.Series
        The two series (x is the leading variable).
    title : str, optional
        Custom plot title.  If None a generic title is generated.
    """
    lag_stats = lag_corr_stats(x, y, max_lag=max_lag)

    if show_table:
        display(lag_stats.style
                .format({'corr': '{:.3f}', 'p_value': '{:.3f}',
                         'ci_low': '{:.3f}', 'ci_high': '{:.3f}',
                         'slope': '{:.3f}', 'n_obs': '{:.0f}'})
                .background_gradient(subset=['corr'], cmap='RdYlGn', vmin=-1, vmax=1))

    if title is None:
        title = (f'Correlation between {x.name or "x"} and {y.name or "y"} '
                 f'({x.index[0].year}–{x.index[-1].year})')

    ax = lag_stats['corr'].plot(figsize=figsize, grid=True,
                                title=title,
                                ylabel=ylabel or 'Coefficient of correlation',
                                label='Pearson corr')
    ax.fill_between(lag_stats.index,
                     lag_stats['ci_low'], lag_stats['ci_high'],
                     alpha=0.25, label='approx. 95 % CI (Fisher z)')
    ax.axhline(0, color='k', lw=0.8)
    ax.legend()

    return lag_stats, ax


def plot_lag_scatter(x: pd.Series, y: pd.Series, lag: int,
                     title_suffix: str = '', figsize: tuple = (8, 6)):
    """
    Scatter of x_t vs y_{t+lag} with OLS regression line.
    Useful to visualise the economic magnitude of the relationship.
    """
    paired = pd.concat([x, y.shift(-lag)], axis=1).dropna()
    xx = paired.iloc[:, 0].values
    yy = paired.iloc[:, 1].values

    slope, intercept = np.polyfit(xx, yy, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xx, yy, alpha=0.55, s=25, edgecolors='none')
    x_line = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2,
            label=f'slope = {slope:.2f}')
    ax.set_xlabel(f'{x.name or "x"} (t)')
    ax.set_ylabel(f'{y.name or "y"} (t + {lag})')
    ax.set_title(f'{x.name or "x"} → {y.name or "y"} (lag = {lag}){title_suffix}')
    ax.legend()
    ax.grid(True)
    return ax
