import numpy as np
import pandas as pd
from math import sqrt

import yfinance as yfin


class Metrics:
    """
    Metrics such as market capitalization, value of stocks traded, turnover ratio to capitalization, volatility.
    """

    ADJ_CLOSE = 'Close'
    VOLUME = 'Volume'
    CAPITALIZATION = 'Capitalization'
    TOTAL_VALUE = 'Total value'
    TURNOVER = 'Turnover ratio'
    VOLATILITY = 'Volatility'
    TRADING_DAYS_IN_YEAR = 252
    TO_ANNUAL_MULTIPLIER = sqrt(TRADING_DAYS_IN_YEAR)

    def __init__(self, tickers, stock_index=None, start=None, end=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a list representing all the ticker symbols making up a stock market that this class
                        derives metrics for
        :param stock_index: an optional ticker symbol of the index that corresponds to the market represented
                            by the 'tickers' parameter
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        """
        self.ticker_symbols = tickers
        self.tickers = yfin.Tickers(self.ticker_symbols)
        self.data = self.tickers.download(start=start, end=end, auto_adjust=True, actions=False, ignore_tz=True)
        self.data = self.data.loc[:, ([Metrics.ADJ_CLOSE, Metrics.VOLUME])]

        # In case some of the stocks captured in the tickers list were not trading during the date range,
        # I assume they were trading using their first price.
        self.data.bfill(inplace=True)

        # Unfortunately only for the most recent trading day
        self.dividend_yield = {}
        for ticker in self.ticker_symbols:
            dividend_yield = self.tickers.tickers[ticker].info.get('dividendYield')
            self.dividend_yield[ticker] = 0. if dividend_yield is None else dividend_yield

        self.shares_outstanding = {}
        for ticker in self.ticker_symbols:
            shares_outst = self.tickers.tickers[ticker].get_shares_full(start=start, end=end).tz_localize(None)
            # Unfortunately Yahoo-Finance occasionally reports duplicate values for shares outstanding
            # for the same date. In such cases I take the most recent value.
            shares_outst = shares_outst.groupby(level=0).last()

            # Yahoo-Finance doesn't report shares outstanding for each trading day. I compensate for it by rolling
            # forward the most recent reported value
            missing_dates = self.data.index.difference(shares_outst.index)
            shares_outst = pd.concat([shares_outst, pd.Series(np.nan, index=missing_dates)]).sort_index()
            shares_outst = shares_outst.ffill().bfill()

            # Getting rid of extraneous dates
            self.shares_outstanding[ticker] = shares_outst.loc[self.data.index]

        self.capitalization = pd.DataFrame(0., index=self.data.index,
                                           columns=[Metrics.CAPITALIZATION, Metrics.TURNOVER])
        self.forward_dividend_yield = 0.
        for ticker in self.ticker_symbols:
            # print('Processing {:s}'.format(ticker))
            df = self.data.loc(axis=1)[:, ticker].droplevel(1, axis=1)
            # Turnover = Closing price x Volume
            self.capitalization.loc[:, Metrics.TURNOVER] += df.iloc[:,0] * df.iloc[:,1]

            # Capitalization = Closing price x Shares outstanding.
            self.capitalization.loc[:, Metrics.CAPITALIZATION] += df.iloc[:,0] * self.shares_outstanding[ticker]

            # Most recent forward dividend yield
            self.forward_dividend_yield += (df.iloc[:,0] * self.shares_outstanding[ticker]).iloc[-1]\
                * self.dividend_yield[ticker]

        self.forward_dividend_yield /= self.capitalization.iloc[-1,0]

        if stock_index is not None:
            self.stock_index_data = yfin.download(
                stock_index, start=start, end=end, auto_adjust=True, actions=False, ignore_tz=True)\
                    .loc[:, Metrics.ADJ_CLOSE]

    def get_capitalization(self, frequency='M', tickers=None):
        """
        Calculates the capitalization of a given market or a subset of stocks over time. Downsamples if a less
        granular frequency than daily is specified. Takes an average capitalization over periods implied by
        the 'frequency' parameter.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the capitalization of the market
        """
        if tickers is None:
            return self.capitalization.loc[:, Metrics.CAPITALIZATION].resample(frequency).mean().dropna()
        else:
            ret = pd.Series(0., index=self.data.index)
            for ticker in tickers:
                ret += self.data.loc[:, (Metrics.ADJ_CLOSE, ticker)] * self.shares_outstanding[ticker]
            return ret.resample(frequency).mean().ret.dropna()

    def get_daily_trading_value_ds(self, tickers=None):
        if tickers is None:
            daily_turnover = self.capitalization.loc[:, Metrics.TURNOVER].copy()
        else:
            daily_turnover = pd.Series(0., index=self.data.index)
            for ticker in tickers:
                daily_turnover += self.data.loc[:, (Metrics.ADJ_CLOSE, ticker)]\
                    * self.data.loc[:, (Metrics.VOLUME, ticker)]
        return daily_turnover

    def get_daily_trading_value(self, frequency='M', tickers=None):
        """
        Calculates a total daily trading value of stocks in the market implied by this object.
        Downsamples if a less granular frequency than daily is specified. Takes an average turnover over periods
        implied by the 'frequency' parameter.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the turnover
        """
        return self.get_daily_trading_value_ds(tickers).resample(frequency).mean().dropna().rename(self.TOTAL_VALUE)

    def get_annual_trading_value(self, frequency='M', tickers=None):
        return self.get_daily_trading_value(frequency, tickers) * self.TRADING_DAYS_IN_YEAR

    def get_daily_turnover(self, frequency='M', tickers=None):
        """
        Calculates a daily turnover ratio of stocks in the market implied by this object to its capitalization.
        Downsamples if a less granular frequency than daily is specified. Takes an average turnover over periods
        implied by the 'frequency' parameter.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the turnover
        """
        daily_turnover = self.get_daily_trading_value_ds(tickers)
        if tickers is None:
            daily_turnover /= self.capitalization.loc[:, Metrics.CAPITALIZATION]
        else:
            capitalization = pd.Series(0., index=self.data.index)
            for ticker in tickers:
                capitalization += self.data.loc[:, (Metrics.ADJ_CLOSE, ticker)] * self.shares_outstanding[ticker]
            daily_turnover /= capitalization

        return daily_turnover.resample(frequency).mean().dropna()

    def get_annual_turnover(self, frequency='M', tickers=None):
        """
        Calculates an annual turnover ratio of stocks in the market implied by this object to its capitalization.
        Downsamples if a less granular frequency than daily is specified. Takes an average turnover over periods
        implied by the 'frequency' parameter.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the turnover
        """
        return self.get_daily_turnover(frequency, tickers) * self.TRADING_DAYS_IN_YEAR

    def get_annual_volatility(self, alpha=1-.94453, frequency='M'):
        """
        Calculates an annual volatility of a market represented by the stock market index used when constructing
        this instance of Metrics. It uses an exponentially weighted moving average (EWMA) with a given smoothing factor
        alpha. It downsamples it if a less granular frequency than daily is specified. When downsampling it takes
        an average volatility over periods implied by the 'frequency' parameter.

        :param alpha: a smoothing factor alpha. Then calculating the EWMA, the most recent observation will be
                      multiplied by 'alpha', while the previous estimate of EMWA by '(1-alpha)'
        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :returns: a pd.Series object capturing the volatility
        """
        if self.stock_index_data is None:
            return None
        vol = self.stock_index_data.pct_change().ewm(alpha=alpha).std() * self.TO_ANNUAL_MULTIPLIER
        return vol.resample(frequency).mean().dropna().rename(self.VOLATILITY)


class USStockMarketMetrics(Metrics):
    def __init__(self, tickers, stock_index='^GSPC', start=None, end=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a list representing all the ticker symbols making up a stock market that this class
                        derives metrics for
        :param stock_index: an optional ticker symbol of the index that corresponds to the market represented
                            by the 'tickers' parameter
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        """
        super().__init__(tickers, stock_index, start, end)

    @staticmethod
    def get_sp500_components():
        """
        Returns the constituent components of the S&P 500 Stock Index. Given that three corporations that are part
        of the index has class B shares, the method returns 503 ticker symbols
        """
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        # Correction for Yahoo-Finance's representation of Class B shares
        sp500_components = [ticker.replace('.', '-') for ticker in df['Symbol'].to_list()]
        return sp500_components


if __name__ == "__main__":
    import sys
    import locale
    from datetime import date

    try:
        locale.setlocale(locale.LC_ALL, '')
        start = date(2022, 1, 1)
        end = date.today()

        sp500_metrics = USStockMarketMetrics(USStockMarketMetrics.get_sp500_components(), start=start, end=end)

        print(sp500_metrics.get_capitalization().tail(10))
        print(sp500_metrics.get_daily_turnover().tail(10))
        print(sp500_metrics.get_annual_turnover().tail(10))
        print(sp500_metrics.get_annual_volatility().tail(10))
        print(f'S&P 500 forward dividend yield on {sp500_metrics.data.index[-1]:%Y-%m-%d} is '
              f'{sp500_metrics.forward_dividend_yield:.3%}')

    except:
        print("Unexpected error: ", sys.exc_info())
