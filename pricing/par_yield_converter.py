import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import math


# class ParYieldConverter:
#     """
#     Converter of par yields to spot rates (aka zero rates) via bootstrapping.
#     """

PAR_VALUE = 100.

#@staticmethod
def par_yields_to_spot(par_yields, maturities, coupon_frequency):
    """
    Converts par yields to spot rates using cubic spline interpolation and bootstrapping.

    Parameters:
    - par_yields (list or array): Par yields (e.g. 0.024 for 2.4%) at corresponding maturities, compounding
                                  assumed to correspond to the coupon_frequency
    - maturities (list or array): Maturities in years
    - coupon_frequency (int): Number of coupon payments per year (e.g. 2 for semiannual)

    Returns:
    - dict: Mapping of maturity (years) to annualized spot rate (as decimal)
    """
    par_yields = np.asarray(par_yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if len(par_yields) != len(maturities):
        raise ValueError("par_yields and maturities must have the same length")

    if coupon_frequency == 0:
        return dict(zip(maturities, par_yields))

    if len(par_yields) == 0:
        raise ValueError("par_yields must be a non-empty sequence")

    cashflow_interval = 1 / coupon_frequency

    if not np.all(maturities[:-1] <= maturities[1:]):
        raise ValueError("maturities must be in ascending order")

    if not np.all((maturities <= cashflow_interval) | (maturities % cashflow_interval == 0)):
        raise ValueError("Each maturity must align with coupon payment frequency")

    # Build full schedule based on cash flow intervals
    all_maturities = np.arange(cashflow_interval, maturities[-1] + cashflow_interval, cashflow_interval)

    # Interpolate par yields using cubic splines
    spline = CubicSpline(maturities, par_yields)
    interpolated_par_yields: np.ndarray = spline(all_maturities)

    # Bootstrapping
    spot_rates = [interpolated_par_yields[0]]

    for i in range(1, len(all_maturities)):
        coupon = interpolated_par_yields[i] / coupon_frequency * PAR_VALUE
        pv_coupons = 0.0

        for j in range(i):
            r = spot_rates[j]
            discount = (1 + r / coupon_frequency) ** (j + 1)
            pv_coupons += coupon / discount

        final_cf = coupon + PAR_VALUE
        remaining_value = PAR_VALUE - pv_coupons
        spot_period = (final_cf / remaining_value) ** (1 / (i + 1)) - 1
        spot_rates.append(spot_period * coupon_frequency)

    output = dict(zip(all_maturities, spot_rates))

    # Patch short-term maturities (zero-coupon case)
    output.update({
        m: y for m, y in zip(maturities, par_yields)
        if m < cashflow_interval
    })

    return output

#@staticmethod
def rates_to_semiannual_yields(rates):
    """
    Convert annually-compounded zero rate(s) → semiannual constant-maturity par yield(s)

    Input:
        - Single float → returns only the 1-year par yield
        - Sequence of N rates [R1, R2, ..., RN] → returns [y1, y2, ..., yN]
        - DataFrame (row-wise) → one row of par yields per input row

    You cannot get a 5-year par yield without the 5-year zero rate (and all prior ones).
    """

    # ------------------------------------------------------------------
    # 1. Core calculation
    # ------------------------------------------------------------------
    def _core_calc(row):
        row = row.values if isinstance(row, pd.Series) else np.asarray(row, dtype=float)
        if len(row) == 0:
            return np.nan

        dfs = [1 / math.sqrt(1 + row[0]), 1 / (1 + row[0])]
        ret = [2 * (math.sqrt(1 + row[0]) - 1)]
        annuity = dfs[0] + dfs[1]

        for i in range(1, len(row)):
            frwd = (1 + row[i]) ** (i + 1) / (1 + row[i - 1]) ** i - 1
            dfs.append(dfs[-1] * (1 + frwd) ** -0.5)
            dfs.append((1 + row[i]) ** -(i + 1))
            annuity += dfs[-1] + dfs[-2]
            ret.append(2 * (1 - dfs[-1]) / annuity)

        return pd.Series(ret, index=row.index if hasattr(row, 'index') else range(1, len(ret) + 1))

    # ------------------------------------------------------------------
    # 2. Convert everything to a pd.Series (the universal container)
    # ------------------------------------------------------------------
    if isinstance(rates, (list, tuple, np.ndarray)):
        rates = pd.Series(rates)
    elif isinstance(rates, pd.DataFrame):
        # Apply row-wise by default (most common in yield curve tables)
        result = rates.apply(_core_calc, axis=1)
        result.name = "Par Yield (semi)"
        return result
    elif not isinstance(rates, pd.Series):
        rates = pd.Series([float(rates)])  # single number

    # If we have a single Series → run once
    result = _core_calc(rates)

    # Preserve name and index
    if isinstance(result, pd.Series):
        result.name = "Par Yield (semi)"

    return result