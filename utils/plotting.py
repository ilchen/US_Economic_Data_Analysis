import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

from pandas.tseries.offsets import DateOffset


def get_recession_periods(recession_series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert a binary recession indicator (e.g. USREC) into list of (start, end) periods.
    """
    recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_recession = False
    start_date = None

    for date, value in recession_series.items():
        if value == 1 and not in_recession:
            start_date = date
            in_recession = True
        elif value == 0 and in_recession:
            recession_periods.append((start_date, date))
            in_recession = False

    if in_recession and start_date is not None:
        recession_periods.append((start_date, recession_series.index[-1]))

    return recession_periods

def add_recession_shading(ax, recessions: list[tuple[pd.Timestamp, pd.Timestamp]], color: str="#cccccc", alpha: float=0.4,
                         label: str = "NBER Recession Periods", add_legend_entry: bool = True) -> None:
    """Add shaded vertical bands for recessions."""
    for start, end in recessions:
        ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
        
    # Add legend entry (only once)
    if add_legend_entry:
        recession_patch = mpatches.Patch(facecolor=color, alpha=alpha, edgecolor="none", label=label)
    
        # Smart merge with existing legend
        handles, labels = ax.get_legend_handles_labels()
        if label not in labels:          # avoid duplicates
            ax.legend(handles=[recession_patch] + handles, labels=[label] + labels, loc="best", frameon=True)

def plot_contributions_waterfall(df, total_growth, title, total_label="US GDP growth YTD", ytd_suffix="",
                                 figsize=(20, 10), ax=None):
    """
    Creates a polished waterfall chart showing contributions to growth.

    Parameters
    ----------
    df : pd.DataFrame
        Must have:
        - Index = category names (industries / components)
        - Column 0 (or 'contribution') = contribution values (positive/negative)
        - Column 'Shifted_Total' = bottom position for each bar (for waterfall stacking)
        Optionally column 1 can be used for connecting lines if different from Shifted_Total.
    total_growth : float
        The final total growth value (for the horizontal reference line).
    title : str
        Main title of the chart.
    total_label : str
        Label for the total growth line.
    ytd_suffix : str
        Suffix for the title (e.g. " (YTD)").
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    positive_color = '#2ca02c'   # vibrant green
    negative_color = '#d62728'   # vivid red

    # Colors for bars
    values = df.iloc[:, 0] if df.shape[1] > 0 else df['contribution']
    colors = [positive_color if x >= 0 else negative_color for x in values]

    # Bars
    bars = ax.bar(df.index, values, bottom=df['Shifted_Total'], width=0.65, color=colors, edgecolor='white', linewidth=1.8, zorder=3)

    # === VALUE LABELS ON BARS ===
    for i, bar in enumerate(bars):
        height = values.iloc[i]
        bottom = df['Shifted_Total'].iloc[i]
        y = bottom + height / 2

        # White text on sufficiently large bars, black otherwise
        text_color = 'white' if abs(height) > 0.012 else 'black'

        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'{height:+.2%}', ha='center', va='center', fontsize=11, fontweight='bold', color=text_color, zorder=5)

    # === CONNECTING LINES (vertical gray dashed) ===
    for i in range(1, len(df)):
        # Use column 1 if it exists and is different, otherwise Shifted_Total
        if df.shape[1] > 1:
            prev_total = df.iloc[i-1, 1]
            curr_total = df.iloc[i, 1]
        else:
            prev_total = df['Shifted_Total'].iloc[i-1]
            curr_total = df['Shifted_Total'].iloc[i]

        ax.plot([i - 0.5, i - 0.5], [prev_total, curr_total], color='gray', linestyle='--', linewidth=1.6, zorder=2)

    # === TOTAL GROWTH REFERENCE LINE ===
    ax.axhline(y=total_growth, linestyle='-', linewidth=3.5, color='#1f77b4', zorder=4)
    ax.text(0, total_growth + 0.0002, total_label + ytd_suffix, size='large', color='#1f77b4')

    # === FORMATTING ===
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=45, ha='right')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title(title + ytd_suffix, fontsize=16, pad=20)

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.margins(x=0.02, y=0.15)

    sns.despine(left=True, bottom=False)

    return fig, ax
