import pandas as pd

def show_monthly(df, rows=None, fmt='%Y-%m'):
    """
    Display a monthly DataFrame with a clean 'Month' index.
    fmt: strftime format for the index labels (default 'YYYY-MM')
    """
    out = df.copy()
    out.index = out.index.strftime(fmt)
    out.index.name = 'Month'
    if rows is not None:
        out = out.tail(rows) if isinstance(rows, int) else out.loc[rows]
    return out


def show_annual(df, rows=None):
    """
    Display an annual / YTD DataFrame with a clean 'Year' index
    showing only the year numbers.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.year
    out.index.name = 'Year'
    if rows is not None:
        out = out.tail(rows) if isinstance(rows, int) else out.loc[rows]
    return out
