from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import seaborn as sns
import scipy.stats


class CapChangePlotter:
    """
    Helper class for plotting capitalization change bars.
    Automatically derives title_prefix and subtitle from the two dates.
    """

    def __init__(
        self,
        start_date: str | datetime | pd.Timestamp | None = None,
        end_date: str | datetime | pd.Timestamp | None = None,
    ):
        """
        start_date, end_date: Will be converted to pd.Timestamp.
        title_prefix and subtitle are derived automatically.
        """
        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp("now")

        # Derive title parts
        self.title_prefix, self.subtitle = self._derive_title_parts()

    def _get_end_label(self, dt: pd.Timestamp) -> str:
        next_bd = dt + pd.offsets.BDay(1)
        ye = pd.offsets.YearEnd().rollforward(dt)
        return "now" if next_bd < ye else f"YE {dt.year}"

    def _get_start_label(self, dt: pd.Timestamp) -> str:
        """Symmetric logic for start date (returns "YE {year}" if it's a year-end)."""
        if dt is None:
            return ""
        # Check if it's exactly a year-end
        ye = pd.offsets.YearEnd().rollforward(dt - pd.offsets.BDay(1))
        if dt == ye:
            return f"YE {dt.year}"
        return dt.strftime("%Y-%m-%d")

    def _derive_title_parts(self) -> tuple[str, str]:
        """Derive both title_prefix and the 'From ... to ...' subtitle."""
        if self.start_date is None or self.end_date is None:
            return "Change in", "From ? to ?"

        start_label = self._get_start_label(self.start_date)
        end_label = self._get_end_label(self.end_date)

        # Main title prefix (exactly what you see in your charts)
        title_prefix = f"{start_label} to {end_label}"

        # Subtitle line
        start_str = self.start_date.strftime("%Y-%m-%d")
        subtitle = f"From {start_str} to {end_label}"

        return title_prefix, subtitle

    def plot_cap_change_bars(self, changes: pd.Series, mode: str='top', cut_off: int = 40,
                             name_expander: Callable[[str], str] = lambda x: x):
        """
        Plot horizontal bar chart for gainers, losers, or mixed.
        """
        if mode == "top":
            data = changes.iloc[-cut_off:]
            label_offset = 1.01
            label_ha = "left"
            label_color = "black"
            cmap_name = "Blues"
        elif mode == "bottom":
            data = changes.iloc[:cut_off]
            label_offset = 1.01
            label_ha = "right"
            label_color = "black"
            cmap_name = "Reds"
        elif mode == "mixed":
            data = changes.iloc[-cut_off:]
            label_offset = 1.01
            label_ha = None
            label_color = "black"
            cmap_name = "RdBu"
        else:
            raise ValueError("mode must be 'top', 'bottom', or 'mixed'")

        display_names = [name_expander(ticker) for ticker in data.index]

        fig_height = max(8, max(len(display_names), cut_off) * 0.42)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        # Color logic (unchanged)
        if mode == "mixed":
            colors = []
            for v in data.values:
                if v >= 0:
                    norm = plt.Normalize(0, data.max())
                    colors.append(plt.get_cmap("Blues")(norm(v)))
                else:
                    norm = plt.Normalize(data.min(), 0)
                    colors.append(plt.get_cmap("Reds")(1 - norm(v)))
        else:
            norm = plt.Normalize(data.min(), data.max())
            cmap_obj = plt.get_cmap(cmap_name)
            if mode == "bottom":
                colors = cmap_obj(1 - norm(data.values))
            else:
                colors = cmap_obj(norm(data.values))

        bars = ax.barh(display_names, data.values, color=colors, edgecolor="white", linewidth=0.5)

        # Labels
        for bar in bars:
            width = bar.get_width()
            if mode == "mixed":
                x = width * 1.01
                ha = 'left' if width >= 0 else 'right'
            else:
                x = width * label_offset
                ha = label_ha
            ax.text(x, bar.get_y() + bar.get_height()/2,
                    f'{width:.1%}',
                    va='center', ha=ha,
                    fontsize=10, fontweight='bold', color=label_color)

        # Titles & styling
        title = f'{self.title_prefix} change in capitalization, %'\
                    + (f' ({mode} {cut_off})\n' if cut_off < len(changes) and mode != 'mixed' else '\n') + self.subtitle
        ax.set_title(title)

        ax.xaxis.set_major_formatter("{x:,.0%}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        ax.set_ylim(-0.5, len(display_names) - 0.5)
        if mode == "bottom":
            ax.invert_yaxis()

        fig.subplots_adjust(left=0.32)
        plt.tight_layout()

        return fig, ax


class BankROEPBPlotter:
    """
    Helper class for plotting Price-to-Book vs ROE scatter for US Banks
    (current / forward / both modes) with regression and cost-of-equity printing.
    """

    def __init__(self, ticker_to_name: dict[str, str] | None = None, metrics=None, geography='US'):
        """
        ticker_to_name: dict for expanding ticker → nice name
        metrics: main metrics object (used for date in title + beta/rfr calc)
        """
        self.ticker_to_name = ticker_to_name or {}
        self.metrics = metrics
        self.geography = geography

    def plot(self, roe_pb_df, roe_cutoff=None, pb_cutoff=None,
             slope_intercept_r_p_value=None, expand_tickers=True, mode: str='current'):
        """
        Plots Price-to-Book vs ROE for Banks.

        mode : {'current', 'forward', 'both'}
            'current'  → only large bubbles with current ROE/PB
            'forward'  → only large bubbles with forward ROE/PB
            'both'     → large bubble (current) + smaller bubble (forward) + arrow
        """
        # === VALIDATION ===
        valid_modes = {"current", "forward", "both"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid options: {valid_modes}")

        assert not (bool(roe_cutoff) ^ bool(pb_cutoff)), "Either both cutoff values or none"

        if roe_cutoff:
            roe_cutoff_min, roe_cutoff_max = roe_cutoff
            roe_pb_df = roe_pb_df.loc[(roe_pb_df.ROE < roe_cutoff_max) & (roe_pb_df.ROE > roe_cutoff_min) & (roe_pb_df["P/B"] < pb_cutoff)]

        # Determine which ROE and P/B to use for the main regression line
        if mode in ['forward', 'both']:
            slope, intercept, rvalue, pvalue = slope_intercept_r_p_value if slope_intercept_r_p_value else \
                            scipy.stats.linregress(roe_pb_df.iloc[:,2].to_list(), roe_pb_df.iloc[:,3].to_list())[0:4]
        else:
            slope, intercept, rvalue, pvalue = slope_intercept_r_p_value if slope_intercept_r_p_value else \
                            scipy.stats.linregress(roe_pb_df.iloc[:,0].to_list(), roe_pb_df.iloc[:,1].to_list())[0:4]

        # Reset index for seaborn
        plot_df = roe_pb_df.reset_index()
        ticker_col = plot_df.columns[0]

        # === SEABORN SCATTER ===
        plt.figure(figsize=(15, 10))
        ax = sns.scatterplot(
            data=plot_df,
            x=plot_df.columns[1] if mode != "forward" else plot_df.columns[3],
            y=plot_df.columns[2] if mode != "forward" else plot_df.columns[4],
            hue=ticker_col,
            palette="tab10",
            s=140,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            legend=False,
        )

        # === FORWARD BUBBLES + ARROWS (mode == 'both') ===
        if mode == "both":
            scatter = ax.collections[0]
            colors = scatter.get_facecolors()
            unique_tickers = plot_df[ticker_col].unique()
            color_dict = dict(zip(unique_tickers, colors))

            for ticker, row in roe_pb_df.iterrows():
                roe_current = row["ROE"]
                pb_current = row["P/B"]
                roe_forward = row.get("Forward ROE")
                pb_forward = row.get("Forward P/B")

                if pd.notna(roe_forward) and pd.notna(pb_forward):
                    color = color_dict.get(ticker)
                    # Forward bubble (smaller, more translucent)
                    ax.scatter(roe_forward, pb_forward, s=80, alpha=0.45, color=color,
                               edgecolor='black', linewidth=1.2, zorder=5)
                    # Pointed dashed arrow
                    ax.annotate('', xy=(roe_forward, pb_forward), xytext=(roe_current, pb_current),
                                arrowprops=dict(arrowstyle='->', color='gray', linestyle='--',
                                                linewidth=1.8, alpha=0.5, shrinkA=6, shrinkB=6))

        # Ticker labels - DYNAMIC offset based on x-axis range
        x_col = "ROE" if mode != "forward" else "Forward ROE"
        x_range = roe_pb_df[x_col].max() - roe_pb_df[x_col].min()
        offset = x_range * 0.005 if x_range > 0 else 0.002

        for ticker, row in roe_pb_df.iterrows():
            name = self.ticker_to_name.get(ticker, ticker) if expand_tickers else ticker
            x_pos = row["ROE"] if mode != "forward" else row["Forward ROE"]
            y_pos = row["P/B"] if mode != "forward" else row["Forward P/B"]
            plt.text(x_pos + offset, y_pos, name, fontsize=10, ha='left', va='bottom', alpha=0.95)

        # Regression line
        x_vals = np.array([roe_pb_df.iloc[:, 0].min(), roe_pb_df.iloc[:, 0].max()]) \
                 if mode not in ['forward', 'both'] else np.array([roe_pb_df.iloc[:, 2].min(), roe_pb_df.iloc[:, 2].max()])
        plt.plot(x_vals, intercept + slope * x_vals, 
                 color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Regression line')

        # Formatting
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))

        # Stats box
        stats_text = (f'Slope = {slope/100:.3f}\n'
                      f'R² = {rvalue**2:.3f}\n'
                      f'p-value = {pvalue:.2g}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                                                   facecolor="white", alpha=0.95, edgecolor="gray"))

        # Titles / labels
        x_label_sfx = {"current": " (ttm)", "forward": " (forward)", "both": " (ttm and forward)"}[mode]
        y_label_sfx = {"current": " (mrq)", "forward": " (forward)", "both": " (mrq and forward)"}[mode]

        title = f"Price to Book vs ROE of {self.geography} Banks"
        if roe_cutoff is not None:
            title += f" (subset: {roe_cutoff_min:.0%} < ROE < {roe_cutoff_max:.0%} and P/B < {pb_cutoff})"
        if self.metrics is not None:
            title += f" on {self.metrics.capitalization.index[-1]:%Y-%m-%d}"

        plt.title(title)
        x_label = roe_pb_df.columns[0] + x_label_sfx
        plt.xlabel(x_label)
        plt.ylabel(roe_pb_df.columns[1] + y_label_sfx)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return roe_pb_df, slope, intercept, rvalue, pvalue

    def print_cost_of_equity(self, roe_pb_df, slope, g, additional_beta=False):
        """Print sensitivity and implied cost of equity."""
        print(f'The sensitivity of P/B to ROE for {self.geography} banks: {slope/100:.2f}\n'
              f"This means that for every percentage point improvement in ROE, the bank's P/B increases by {slope/100:.2f}\n")
        print(f'From the above equation it also means that an approximate cost of equity for these banks is: {1./slope + g:.2%}')
        print(roe_pb_df.index.to_list())

        if self.metrics and additional_beta:
            rfr = self.metrics.riskless_rate.iloc[-1]
            coe = roe_pb_df.index.to_series().map(
                lambda ticker: (self.metrics.tickers.tickers[ticker].info.get("beta")
                    if (beta := self.metrics.tickers.tickers[ticker].info.get("beta")) is not None and beta >= 0.1
                    else self.metrics.get_beta([ticker]) ) * 0.05 + rfr).mean()
            print(f"When calculating cost of equity based on their beta to the market, it becomes {coe:.2%}")
