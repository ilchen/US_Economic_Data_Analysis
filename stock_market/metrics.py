import numpy as np
import pandas as pd
from pandas.tseries.offsets import YearBegin, BDay, MonthBegin
import pandas_datareader.data as web
from math import sqrt, isclose
from datetime import date, time, datetime
import os
import warnings

import yfinance as yfin


class Metrics:
    """
    Metrics such as market capitalization, value of stocks traded, turnover ratio to capitalization, volatility.
    """

    ADJ_CLOSE = 'Close'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    CAPITALIZATION = 'Capitalization'
    TOTAL_VALUE = 'Total value'
    TURNOVER = 'Turnover ratio'
    MKT_SHARE = 'Market share (%)'
    VOLATILITY = 'Volatility'
    TRADING_DAYS_IN_YEAR = 252
    TO_ANNUAL_MULTIPLIER = sqrt(TRADING_DAYS_IN_YEAR)

    def __init__(self, tickers, additional_share_classes=None, stock_index=None, start=None, hist_shares_outs=None,
                 tickers_to_correct_for_splits=None, currency_conversion_df=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a dictionary representing all the ticker symbols making up a stock market that this class
                        derives metrics for. Each key represents a ticker symbol. Each value designates the start and
                        end day of the stock represented by the ticker being added or removed. An end value of None
                        implies it's still part of the market, a start value of None designates it becoming part of
                        the market before 'start'
        :param additional_share_classes: a dictionary whose keys are additional share classes of companies
                                         that have multiple share classes and where they all are part of the market,
                                         the values are the first share class (typically class A)
        :param stock_index: an optional ticker symbol of the index that corresponds to the market represented
                            by the 'tickers' parameter
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses different kinds of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param hist_shares_outs: a dictionary representing historical shares outstanding for delisted tickers.
                            Each key represents a ticker symbol. Each value is a panda Series designating shares
                            outstanding on certain days.
        :param tickers_to_correct_for_splits: Yahoo-Finance sometimes incorrectly reports close price by automatically
                                              adjusting it for splits, this list contains ticker symbols for which
                                              a correction is required.
        :param currency_conversion_df: for heterogeneous markets made up of stocks priced in different currencies,
                                       a DataFrame whose columns are suffixes such as '.L', '.SW', '.CO', etc. and whose
                                       rows are the corresponding conversion rates.
        """
        self.ticker_symbols = tickers
        self.tickers = yfin.Tickers(list(self.ticker_symbols.keys()))
        # When calculating market capitalization and trading volumes, we need to use nominal close prices
        # and not adjusted close prices
        self.data = self.tickers.download(start=start, auto_adjust=False, actions=False, ignore_tz=True)
        self.data = self.data.loc[:, ([Metrics.CLOSE, Metrics.VOLUME])]

        # Required until the 'ignore_tz' parameter in the 'download' method starts working again
        self.data.index = self.data.index.tz_localize(None)

        # In case some stocks captured in the tickers list were not trading during the date range,
        # I assume they were trading using their first price.
        self.data.bfill(inplace=True)

        # Unfortunately Yahoo-Finance provides dividend yield only for the most recent trading day
        self.dividend_yield = {}
        self.pe = {}
        for ticker in self.get_current_components():
            info = self.tickers.tickers[ticker].info
            dividend_yield = info.get('dividendYield')
            pe = info.get('forwardPE')

            # A kludge for London Stock Exchange, Yahoo-Finance sometimes reports incorrect forwardPEs
            sfx = self.get_exchange_suffix(ticker)
            if sfx == '.L':
                trailing_pe = info.get('trailingPE')
                if trailing_pe is not None and trailing_pe / pe < .05:
                    trailing_eps = info.get('trailingEps')
                    eps = info.get('forwardEps')
                    if trailing_eps is not None and eps is not None:
                        pe = trailing_pe * trailing_eps / eps

            self.dividend_yield[ticker] = 0. if dividend_yield is None else dividend_yield
            self.pe[ticker] = 0. if pe is None else pe

        # Inplace setting is perfectly fine
        warnings.filterwarnings('ignore', message='In a future version, `df.+` will attempt to set the values inplace',
                                category=FutureWarning)
        subset = self.data.loc[:, (self.CLOSE,)]
        delisted_tickers = subset.columns[subset.isna().all()]
        for delisted_ticker in delisted_tickers:
            if os.path.isfile(os.path.expanduser(f'./stock_market/historical_equity_data/{delisted_ticker}.cvs')):
                f = pd.read_csv(f'./stock_market/historical_equity_data/{delisted_ticker}.cvs', index_col=0)
            else:
                # Filling in the gaps with Alpha Vantage API for close prices and volumes of delisted shares
                # Please request your own API Key in order to make the below call work
                print(f'About to download historical daily closing prices for {delisted_ticker}')
                f = web.DataReader(delisted_ticker, 'av-daily', start=start, api_key=os.getenv('ALPHAVANTAGE_API_KEY'))
                f.to_csv(f'./stock_market/historical_equity_data/{delisted_ticker}.cvs')

            f = f.set_axis(pd.DatetimeIndex(f.index, self.data.index.freq))
            f = f.loc[self.data.index[0]:, ['close', 'volume']]
            f.columns = pd.MultiIndex.from_tuples(list(zip([self.CLOSE, self.VOLUME], [delisted_ticker]*2)))
            self.data.loc[f.index, ([self.CLOSE, self.VOLUME], delisted_ticker)] = f

        # Walk through all tickers that are no longer part of S&P 500 and fill in missing prices (if any) for
        # those that have been delisted
        for delisted_ticker in [k for (k, (st, ed)) in tickers.items() if ed is not None]:
            ed = min(tickers[delisted_ticker][1], self.data.index[-1])
            f = self.data.loc[:, (self.CLOSE, delisted_ticker)]
            st = f.last_valid_index()
            # In case delisting took place before a formal removal from the market, I replicate the last closing price
            # and assign a volume of 0 to the trading days leading up to the delisting
            if st < ed:
                self.data.loc[st + BDay(1):ed, ([self.CLOSE, self.VOLUME], delisted_ticker)] = (f.loc[st], 0.)

        if tickers_to_correct_for_splits is not None:
            # Ensure we don't bother with tickers that are not part of the market
            tickers_to_correct_for_splits = set(tickers_to_correct_for_splits) & self.ticker_symbols.keys()
            for ticker in tickers_to_correct_for_splits:
                if os.path.isfile(os.path.expanduser(f'./stock_market/historical_equity_data/{ticker}.cvs')):
                    f = pd.read_csv(f'./stock_market/historical_equity_data/{ticker}.cvs', index_col=0)
                else:
                    f = web.DataReader(ticker, 'av-daily', start=start, api_key=os.getenv('ALPHAVANTAGE_API_KEY'))
                    f.to_csv(f'./stock_market/historical_equity_data/{ticker}.cvs')

                f = f.set_axis(pd.DatetimeIndex(f.index, self.data.index.freq))
                f = f.loc[self.data.index[0]:, ['close', 'volume']]
                f.columns = pd.MultiIndex.from_tuples(list(zip([self.CLOSE, self.VOLUME], [ticker]*2)))

                self.data.loc[self.data.index[0]:f.index[-1], ([self.CLOSE, self.VOLUME], ticker)] = f

        # Currency conversion
        if currency_conversion_df is not None:
            for ticker in self.ticker_symbols.keys():
                sfx = self.get_exchange_suffix(ticker)
                if sfx in currency_conversion_df.columns:
                    self.data.loc[:, (self.CLOSE, ticker)] *= currency_conversion_df.loc[self.data.index, sfx]

        self.shares_outstanding = {}
        for ticker in self.ticker_symbols.keys():

            # For delisted shares, Yahoo-Finance doesn't report any data
            if ticker in delisted_tickers:
                if hist_shares_outs is None or ticker not in hist_shares_outs:
                    raise ValueError(f'No data on shares outstanding for {ticker}')
                shares_outst = hist_shares_outs[ticker]

            # In case the historical shares outstanding dictionary has an entry, use it in preference to
            # Yahoo-Finance's API
            # elif hist_shares_outs is not None and ticker in hist_shares_outs \
            #         and (tickers_to_correct_for_splits is None or ticker not in tickers_to_correct_for_splits):
            #     shares_outst = hist_shares_outs[ticker]

            else:
                shares_outst = self.tickers.tickers[ticker].get_shares_full(start=start).tz_localize(None)
                # Unfortunately Yahoo-Finance occasionally reports duplicate values for shares outstanding
                # for the same date. In such cases I take the most recent value.
                shares_outst = shares_outst.groupby(level=0).last()

                # Correction for the shares outstanding for companies that have multiple classes of shares that
                # are listed, e.g. Alphabet's Class A 'GOOGL' and Class C 'GOOG' stocks or 'BRK-A' and 'BRK-B'
                shares_outstanding = self.tickers.tickers[ticker].info.get('sharesOutstanding')
                if shares_outstanding is not None and shares_outstanding * 1.2 < shares_outst.iloc[-1].item():
                    shares_outstanding2 = self.tickers.tickers[ticker].info.get('impliedSharesOutstanding')
                    if shares_outstanding2 is None:
                        shares_outstanding2 = shares_outst.iloc[-1].item()
                    print('Correcting the number of shares outstanding for {:s} from {:d} to {:d}'
                          .format(ticker, shares_outst.iloc[-1].item(),
                                  int(shares_outst.iloc[-1].item() * shares_outstanding / shares_outstanding2)))

                    shares_outst *= shares_outstanding / float(shares_outstanding2)
                    shares_outst = shares_outst.astype('int64')

                # Correcting for shares outstanding based ont he override in hist_shares_outs
                if hist_shares_outs is not None and ticker in hist_shares_outs:
                    correction = hist_shares_outs[ticker]
                    missing_dates_back = shares_outst.loc[:correction.index[-1]].index.difference(correction.index)
                    missing_dates_back = missing_dates_back.union(
                        self.data.loc[:correction.index[-1]].index.difference(correction.index))
                    correction = pd.concat([correction, pd.Series(np.nan, index=missing_dates_back)]).sort_index()
                    correction = correction.bfill()
                    shares_outst.update(correction)
                    print('\tExtra correction for the number of shares outstanding for {:s} for period '
                          'from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(ticker, self.data.index[0], correction.index[-1]))

            # Yahoo-Finance doesn't report shares outstanding for each trading day. I compensate for it by rolling
            # forward the most recent reported value and then rolling backward the earliest available number.
            missing_dates = self.data.index.difference(shares_outst.index)
            shares_outst = pd.concat([shares_outst, pd.Series(np.nan, index=missing_dates)]).sort_index()
            shares_outst = shares_outst.ffill().bfill()

            # Getting rid of extraneous dates
            self.shares_outstanding[ticker] = shares_outst.loc[self.data.index]

        warnings.filterwarnings('ignore', message='DataFrame is highly fragmented',
                                category=pd.errors.PerformanceWarning)
        self.capitalization = pd.DataFrame(0., index=self.data.index,
                                           columns=[Metrics.CAPITALIZATION, Metrics.TURNOVER])
        self.forward_dividend_yield = 0.
        self.forward_PE = 0.
        for ticker in self.ticker_symbols.keys():
            # print('Processing {:s}'.format(ticker))
            # df's first column is the closing price and the second is the volume
            df = self.data.loc(axis=1)[:, ticker].droplevel(1, axis=1)

            (st, ed) = self.ticker_symbols[ticker]
            # Turnover = Closing price x Volume
            self.capitalization.loc[st:ed, Metrics.TURNOVER]\
                += df.loc[st:ed,Metrics.CLOSE] * df.loc[st:ed,Metrics.VOLUME]

            # Capitalization = Closing price x Shares outstanding.
            self.capitalization.loc[st:ed, Metrics.CAPITALIZATION]\
                += df.loc[st:ed,Metrics.CLOSE] * self.shares_outstanding[ticker].loc[st:ed]
            self.capitalization.loc[st:ed, ticker]\
                = df.loc[st:ed,Metrics.CLOSE] * self.shares_outstanding[ticker].loc[st:ed]

            # Most recent forward dividend yield and forward P/E
            if ticker in self.get_current_components():
                self.forward_dividend_yield += (df.iloc[:,0] * self.shares_outstanding[ticker]).iloc[-1]\
                    * self.dividend_yield[ticker]
                self.forward_PE += (df.iloc[:,0] * self.shares_outstanding[ticker]).iloc[-1] \
                    * self.pe[ticker]

        self.forward_dividend_yield /= self.capitalization.iloc[-1,0]
        self.forward_PE /= self.capitalization.iloc[-1,0]

        # Given that a stock index is used for calculating volatility, we need to use adjusted close prices.
        self.stock_index_data = stock_index
        if stock_index is not None:
            # Handy to get earlier data for the index for more accurate estimate of volatility
            self.stock_index_data = yfin.download(
                stock_index, start=start - pd.DateOffset(years=3), auto_adjust=True, actions=False, ignore_tz=True)\
                    .loc[:, (Metrics.ADJ_CLOSE, stock_index)]
            #        .loc[:, Metrics.ADJ_CLOSE], worked in older versions of yfinance
            
            # Required until the 'ignore_tz' parameter in the 'download' method starts working again
            self.stock_index_data = self.stock_index_data.tz_localize(None)
        
        # Must be initialized in a subclass
        self.riskless_rate = None
        
        # Required for the 'get_quarterly_stmt_data' and get_top_n_capitalization_for_x methods 
        if additional_share_classes is None:
            additional_share_classes = {}
        self.additional_share_classes = additional_share_classes

    def get_capitalization(self, frequency='ME', tickers=None):
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
                (st, ed) = self.ticker_symbols[ticker]
                ret += self.data.loc[st:ed, (Metrics.CLOSE, ticker)] * self.shares_outstanding[ticker].loc[st:ed]
            return ret.resample(frequency).mean().ret.dropna()

    def adjust_for_additional_share_classes(self, cap_series):
        for additional_share_class, main_share_class in self.additional_share_classes.items():
            cap_series.loc[main_share_class] += cap_series.loc[additional_share_class]
            cap_series.drop(additional_share_class, inplace=True)

    def get_top_n_capitalization_companies_for_day(self, n, dt=None, merge_additional_share_classes=True):
        """
        Calculates the top-n capitalization companies for the market.

        :param n: the number of maximum capitalization companies to return
        :param dt: (string, int, date, datetime, Timestamp) – the date for which to return the top-n capitalization,
                   if None, the most recent business day is used
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.Series object capturing the top-n capitalization companies for the market along with the
                  percentage share of their capitalization in the market. The pd.Series is indexed by the ticker
                  symbols.
        """
        if dt is None:
            dt = self.data.index[-1]
        idx = -1
        dt = BDay(0).rollback(dt) if BDay(0).rollback(dt) <= self.data.index[idx] else self.data.index[idx]
        while np.isnan(self.capitalization.loc[dt, self.CAPITALIZATION]):
            idx -= 1
            dt = self.data.index[idx]

        cap_series = self.capitalization.loc[dt, self.capitalization.columns[2:]]
        if merge_additional_share_classes and len(self.additional_share_classes) > 0:
            cap_series = cap_series.copy()
            self.adjust_for_additional_share_classes(cap_series)

        topn = cap_series.nlargest(n).to_frame(self.CAPITALIZATION)
        topn[self.MKT_SHARE] = topn.iloc[:, 0] / self.capitalization.loc[dt, self.CAPITALIZATION]

        return topn

    def get_top_n_capitalization_companies_for_month(self, n, dt=None, merge_additional_share_classes=True):
        """
        Calculates the top-n capitalization companies for the market taking the average for the month specified.

        :param n: the number of maximum capitalization companies to return
        :param dt: (string, int, date, datetime, Timestamp) – the date for whose month to return the topn capitalization
                    if None, the most recent business day is used
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.Series object capturing the top-n capitalization companies for the market along with the
                  percentage share of their capitalization in the market. The pd.Series is indexed by the ticker
                  symbols.
        """
        if dt is None:
            dt = self.data.index[-1]
        dt = MonthBegin(0).rollback(dt) if MonthBegin(0).rollback(dt) <= self.data.index[-1]\
            else MonthBegin(0).rollback(self.data.index[-1])
        resampled_cap = self.capitalization.resample('MS').mean()
        cap_series = resampled_cap.loc[dt, resampled_cap.columns[2:]]
        if merge_additional_share_classes and len(self.additional_share_classes) > 0:
            self.adjust_for_additional_share_classes(cap_series)
        topn = cap_series.nlargest(n).to_frame(self.CAPITALIZATION)
        topn[self.MKT_SHARE] = topn.iloc[:, 0] / resampled_cap.loc[dt, self.CAPITALIZATION]
        return topn

    def get_top_n_capitalization_companies_for_year(self, n, dt=None, merge_additional_share_classes=True):
        """
        Calculates the top-n capitalization companies for the market taking the average for the year specified.

        :param n: the number of maximum capitalization companies to return
        :param dt: (string, int, date, datetime, Timestamp) – the date for whose year to return the topn capitalization
                    if None, the current year is used
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.Series object capturing the top-n capitalization companies for the market along with the
                  percentage share of their capitalization in the market. The pd.Series is indexed by the ticker
                  symbols.
        """
        if dt is None:
            dt = self.data.index[-1]
        dt = YearBegin(0).rollback(dt) if YearBegin(0).rollback(dt) <= self.data.index[-1]\
            else YearBegin(0).rollback(self.data.index[-1])
        resampled_cap = self.capitalization.resample('YS').mean()
        cap_series = resampled_cap.loc[dt, resampled_cap.columns[2:]]
        if merge_additional_share_classes and len(self.additional_share_classes) > 0:
            self.adjust_for_additional_share_classes(cap_series)
        topn = cap_series.nlargest(n).to_frame(self.CAPITALIZATION)
        topn[self.MKT_SHARE] = topn.iloc[:, 0] / resampled_cap.loc[dt, self.CAPITALIZATION]
        return topn

    def get_daily_trading_value_ds(self, tickers=None):
        if tickers is None:
            daily_turnover = self.capitalization.loc[:, Metrics.TURNOVER].copy()
        else:
            daily_turnover = pd.Series(0., index=self.data.index)
            for ticker in tickers:
                (st, ed) = self.ticker_symbols[ticker]
                daily_turnover += self.data.loc[st:ed, (Metrics.CLOSE, ticker)]\
                    * self.data.loc[st:ed, (Metrics.VOLUME, ticker)]
        return daily_turnover

    def get_daily_trading_value(self, frequency='ME', tickers=None):
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

    def get_annual_trading_value(self, frequency='ME', tickers=None):
        return self.get_daily_trading_value(frequency, tickers) * self.TRADING_DAYS_IN_YEAR

    def get_daily_turnover(self, frequency='ME', tickers=None):
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
                (st, ed) = self.ticker_symbols[ticker]
                capitalization += self.data.loc[st:ed, (Metrics.CLOSE, ticker)]\
                    * self.shares_outstanding[ticker].loc[st:ed]
            daily_turnover /= capitalization

        return daily_turnover.resample(frequency).mean().dropna()

    def get_annual_turnover(self, frequency='ME', tickers=None):
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

    def get_annual_volatility(self, alpha=1-.94453, frequency='ME'):
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
        st = MonthBegin(0).rollback(self.data.index[0])
        return vol.resample(frequency).mean().dropna().rename(self.VOLATILITY).loc[st:]

    def get_current_components(self):
        """
        Returns the current set of components in a given stock market.
        """
        return [k for k, (st, ed) in self.ticker_symbols.items() if ed is None]

    def get_beta(self, tickers, weights=None, years=5, use_adjusted_close=False):
        """
        Calculates the Capital Asset Pricing Model beta of a group of stocks represented by 'tickers' relative to the
        market portfolio represented by 'self.stock_index_data', if any. It uses prices at the beginning of each
        month and goes back to min('years', start-date-used-to-construct-this-object). When multiple tickers are
        specified, assumes an equal allocation of funds to each stock unless overridden with the 'weights' parameter.

        :param tickers: a list of one or more ticker symbols
        :param weights: a list of float numbers summing up to 1. If more than one ticker symbol is specified, it
                        determines how much to allocate to a particular stock in a portfolio made up of 'tickers'
        :param years: an integer designating how many years in the past to go to calculate the beta of the portfolio
                      made up of 'tickers'
        :param use_adjusted_close: indicates whether to use adjusted closing prices, which produces more accurate
                                   results at the expense of extra network calls to Yahoo-Finance APIs
        :returns: a pd.Series object capturing the beta or None if self.stock_index_data
        """
        if self.stock_index_data is None:
            return None
        n = len(tickers)
        st = MonthBegin(0).rollback(self.stock_index_data.index[-1] - pd.DateOffset(years=years))
        if use_adjusted_close:
            tickers = yfin.Tickers(tickers) if n > 1 else yfin.Ticker(tickers[0])
            # When calculating the Beta of a portfolio relative to the market, it's better to use adjusted close prices
            # and not adjusted close prices
            if n > 1:
                portfolio = tickers.download(start=MonthBegin(0).rollback(self.data.index[0]),
                                             auto_adjust=True, actions=False, ignore_tz=True)
                # Required until the 'ignore_tz' parameter in the 'download' method starts working again
                portfolio.index = portfolio.index.tz_localize(None)
            else:
                portfolio = tickers.history(start=MonthBegin(0).rollback(self.data.index[0]),
                                            auto_adjust=True, actions=False)
                portfolio.index = portfolio.index.tz_localize(None)
            portfolio = portfolio.loc[:, Metrics.CLOSE]
        else:
            portfolio = self.data.loc[:, (Metrics.CLOSE, tickers)]
        if n > 1:
            if weights is None:
                weights = pd.Series([1. / n] * n, index=portfolio.columns)
            else:
                if len(weights) != n or not isclose(sum(weights), 1.):
                    raise ValueError('Weights are not specified correctly')
                if not isinstance(weights, pd.Series):
                    weights = pd.Series(weights, index=portfolio.columns)
                else:
                    weights.index = portfolio.columns
            weights = weights.div(portfolio.iloc[0, :])
            portfolio = portfolio.mul(weights, axis=1)
            portfolio = portfolio.sum(axis=1)
        portfolio = portfolio.resample('MS').first().pct_change().loc[st:].dropna().squeeze()
        market = self.stock_index_data.resample('MS').first().pct_change().loc[st:].dropna()
        common_index = portfolio.index.intersection(market.index)
        portfolio = portfolio.loc[common_index[0]:]
        market = market.loc[common_index[0]:]
        return market.cov(portfolio) / market.var()
    
    def get_excess_return_helper(self, years=2, frequency='ME', sharpe_ratio=False):
        """
        Calculates either an excess return of the marker over the riskless rate or ex-post Sharpe ratio
        of the market represented by this object.

        :param years: an integer designating how many years in the past to go to calculate the volatility of the market
                      that this object represents
        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases must
            be limited to ME, BME, MS, or BMS.
        :param sharpe_ratio: a boolean indicating what to return: if False calculate excess returns, if True the
                             Sharpe ratio
        :returns: a pd.Series object capturing the Sharpe ratio of the market represented by this object
        """
        if self.stock_index_data is None or self.riskless_rate is None:
            return None
        if frequency not in ['ME', 'BME', 'MS', 'BMS']:
            raise ValueError(f'Frequency {frequency} is not supported')
        if frequency in ['MS', 'BMS']:
            riskless_rate = self.riskless_rate.resample(frequency).first()
            market = self.stock_index_data.resample(frequency).first()
        else:
            riskless_rate = self.riskless_rate.resample(frequency).last()
            market = self.stock_index_data.resample(frequency).last()
        market = market.pct_change(12).dropna()
        riskless_rate = riskless_rate.shift(12).dropna()
        volatility = market.rolling(12 * years).std().dropna()

        common_index = volatility.index.intersection(market.index).intersection(riskless_rate.index)
        st = max(MonthBegin(0).rollback(self.data.index[0]), common_index[0])
        volatility = volatility.loc[st:]
        market = market.loc[st:]
        riskless_rate = riskless_rate.loc[st:]
        return (market - riskless_rate) / volatility if sharpe_ratio else market - riskless_rate

    def get_excess_return(self, years=2, frequency='ME'):
        """
        Calculates ex-post excess annual return of the market represented by this object over 1-year riskless rate.
        """
        return self.get_excess_return_helper(years, frequency, False)

    def get_sharpe_ratio(self, years=2, frequency='ME'):
        """
        Calculates ex-post Sharpe ratio of the market represented by this object.

        :param years: an integer designating how many years in the past to go to calculate the volatility of the market
                      that this object represents
        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases must
            be limited to M, BM, MS, or BMS.
        :returns: a pd.Series object capturing the Sharpe ratio of the market represented by this object
        """
        return self.get_excess_return_helper(years, frequency, True)

    def get_roe_and_pb(self, tickers):
        """
        Constructs a DataFrame indexed by 'tickers' and whose columns represent the current ROE (TTM) and Price to Book
        of the corresponding tickers.

        :returns: a pd.DataFrame object capturing the Sharpe ratio the ROE and Price to Book ratio for each index, it
                  may contain NaN values in case Yahoo-Finance didn't have the data
        """
        ret = pd.DataFrame(index=tickers, columns=['ROE', 'P/B'])
        for ticker in tickers:
            if all(k in self.tickers.tickers[ticker].info for k in ('returnOnEquity', 'priceToBook')):
                ret.loc[ticker, 'ROE'] = self.tickers.tickers[ticker].info['returnOnEquity']
                ret.loc[ticker, 'P/B'] = self.tickers.tickers[ticker].info['priceToBook']
                if ticker.endswith('.L'):
                    ret.loc[ticker, 'P/B'] /= 100.
        return ret

    @staticmethod
    def get_exchange_suffix(ticker):
        comps = ticker.split('.')
        sfx = '.' + comps[-1]
        return None if len(comps) == 1 else sfx


class USStockMarketMetrics(Metrics):
    def __init__(self, tickers, additional_share_classes=None, stock_index='^GSPC', start=None, hist_shares_outs=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a dictionary representing all the ticker symbols making up a stock market that this class
                        derives metrics for. Each key represents a ticker symbol, each value the start and end day of
                        the stock represented by the ticker being added or removed. An end value of None implies it's
                        still part of the market, a start value of None designates it becoming part of the market
                        before 'start'
        :param additional_share_classes: a dictionary whose keys are additional share classes of companies
                                         that have multiple share classes and where they all are part of the market,
                                         the values are the first share class (typically class A)
        :param stock_index: an optional ticker symbol of the index that corresponds to the market represented
                            by the 'tickers' parameter
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many kinds of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param hist_shares_outs: a dictionary representing historical shares outstanding for delisted tickers.
                            Each key represents a ticker symbol. Each value is a panda Series designating shares
                            outstanding on certain days.
        """
        # Unfortunately Yahoo-Finance reports incorrect closing prices for shares before they had a stock split
        # or an implicit stock split when a firm spins off one of its units into an independent entity, e.g. recent
        # IBM and T splits.
        super().__init__(tickers, additional_share_classes, stock_index, start, hist_shares_outs,
                         ['GOOG', 'GOOGL', 'AMZN', 'AAPL', 'NDAQ', 'AIV', 'ANET', 'TECH', 'COO', 'NVDA', 'TSLA', 'CPRT',
                          'CSGP', 'CSX', 'DXCM', 'EW', 'FTNT', 'ISRG', 'MNST', 'NEE', 'PANW', 'SHW', 'WMT', 'GE', 'LH',
                          'ODFL', 'MCHP', 'APH', 'DTE', 'FTV', 'MTCH', 'MKC', 'MRK', 'PFE', 'RJF', 'RTX', 'ROL', 'TT',
                          'SLG', 'FTI', 'NVDA', 'CMG', 'AVGO', 'WRB', 'EXC', 'BWA', 'K', 'IP', 'O', 'PCAR', 'DHR',
                          'BBWI', 'BDX', 'ZBH', 'SRE', 'MMM', 'IBM', 'T', 'CTAS', 'DECK', 'SMCI', 'LRCX', 'J'])

        # Using Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, as proxy for riskless rate
        # Handy to get earlier data for more accurate estimates of volatility
        self.riskless_rate = web.get_data_fred('DGS1', start - pd.DateOffset(years=3))

        # Convert into pd.Series and percentage points
        self.riskless_rate = self.riskless_rate.dropna().iloc[:,0] / 100.

    @staticmethod
    def get_sp500_components():
        """
        Returns a pair whose first component is a list capturing the current constituent components of
        the S&P 500 Stock Index and whose second component is a dictionary of companies in S&P500 that have multiple
        share classes as part of the index, the keys are additional shares tickers ticker symbols and values
        are the first share class (typically class A). Three corporations in the index have class B or class C shares.
        """
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        # Correction for Yahoo-Finance's representation of Class B shares
        sp500_components = [ticker.replace('.', '-') for ticker in df['Symbol'].to_list()]
        # additional_share_classes = df.loc[df.loc[:, 'CIK'].duplicated(), 'Symbol'].to_list()
        dict = df.loc[df.loc[:, 'CIK'].duplicated(keep=False), ['Symbol', 'CIK']].groupby('CIK')\
                ['Symbol'].apply(list).to_dict()
        additional_share_classes = {}
        for cik, share_classes in dict.items():
            # The list of share_classes is guaranteed to have more than one value
            main_share_class = share_classes[0]
            for additional_share_class in share_classes[1:]:
                additional_share_classes[additional_share_class] = main_share_class
        # additional_share_classes
        # {'NWS': 'NWSA', 'GOOG': 'GOOGL', 'FOX': 'FOXA'}

        return sp500_components, additional_share_classes

    @staticmethod
    def get_sp500_banking_sector_components():
        """
        Returns a list of ticker symbols of S&P 500 Stock Index companies that belong to the banking sector.
        """
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        # Correction for Yahoo-Finance's representation of Class B shares
        cps = df.loc[(df['GICS Sector'] == 'Financials') & (df['GICS Sub-Industry'].str.find('Banks') != -1), 'Symbol']
        return [ticker.replace('.', '-') for ticker in cps.to_list()]

    @staticmethod
    def get_sp500_historical_components(start=None):
        """
        Returns a dictionary whose keys are ticker symbols representing companies that were part of the S&P 500 Index
        at any time since 'start' and whose values are pairs representing the dates of inclusion and exclusion from
        the index. A start date of 'None' means the ticker was part of the index before 'start'. An end date of 'None'
        implies the ticker is still part of the index.

        :param start: an int, float, str, datetime, or date object designating the starting time for calculating the
                      history of the constituent components of the S&P 500 Index.
        """
        if type(start) is date:
            start = datetime.combine(start, time())
        start = pd.to_datetime(start)
        all_components = USStockMarketMetrics.get_sp500_components()[0].copy()
        ret = {ticker: (start, None) for ticker in all_components}
        removed_tickers = set()

        df = pd.read_csv('./stock_market/sp500_changes_since_2019.csv', index_col=[0])
        for idx, row in df[::-1].iterrows():
            ts = pd.to_datetime(idx)
            if ts < start:
                break
            for added_ticker in [] if pd.isnull(row.iloc[0]) else row.iloc[0].split(','):
                _, end = ret[added_ticker]
                ret[added_ticker] = (ts, end)

            for removed_ticker in [] if pd.isnull(row.iloc[1]) else row.iloc[1].split(','):
                ret[removed_ticker] = (start, ts-BDay(1))
                removed_tickers.add(removed_ticker)
                all_components.append(removed_ticker)

        if len(all_components) > len(ret):
            raise ValueError('Some tickers were added twice during the implied period')

        return ret

    @staticmethod
    def get_sp500_historical_shares_outstanding():
        """
        Returns a dictionary whose keys are ticker symbols representing companies that were part of the S&P 500 Index
        at any time since of 2020 and whose values are pd.Series objects representing the number of shares outstanding
        on a given reporting date. I obtained the numbers by going through the quarterly and annual reports of the
        respective companies.
        """
        last_bd = BDay(0).rollback
        return {'ABMD': pd.Series([45062665, 44956959, 45047262, 45189883, 45230966, 45295979, 45380233, 45497490,
                                   45516200, 45563937, 45460884, 45091184],
                                  index=pd.DatetimeIndex(['2020-01-30', '2020-05-14', '2020-07-30', '2020-10-22',
                                                          '2021-01-25', '2021-05-14', '2021-07-30', '2021-10-21',
                                                          '2022-01-27', '2022-05-13', '2022-07-29', '2022-10-28'])
                                  .map(last_bd)),
                'AGN': pd.Series([329002015, 329805791],
                                 index=pd.DatetimeIndex(['2020-02-12', '2020-05-01']).map(last_bd)),
                'ALXN': pd.Series([221400872, 220827431, 218845432, 219847960, 221019230],
                                  index=pd.DatetimeIndex(['2020-01-29', '2020-05-04', '2020-10-27', '2021-02-12',
                                                          '2021-04-21']).map(last_bd)),
                'ATVI': pd.Series([769221524, 770485455, 772857185, 777016759, 782306592, 782625319, 784274126,
                                   786158727, 786798320],
                                  index=pd.DatetimeIndex(['2020-02-20', '2020-04-28', '2020-10-22', '2021-04-27',
                                                          '2022-07-25', '2022-10-31', '2023-02-16', '2023-04-28',
                                                          '2023-07-24']).map(last_bd)),
                'BRK-B': pd.Series([1385994959, 1401356454, 1390707370, 1370951744, 1336348609, 1326572128, 1325373100,
                                    1303476707, 1291212661, 1285751332, 1301126370, 1301981370, 1301100243, 1295970861,
                                    1308070268],
                                   index=pd.DatetimeIndex(['2020-02-13', '2020-07-30', '2020-08-23', '2020-10-26',
                                                           '2021-02-16', '2021-04-22', '2021-07-26', '2021-10-27',
                                                           '2022-02-14', '2022-04-20', '2022-07-26', '2022-10-26',
                                                           '2023-02-13', '2023-04-25', '2023-07-26']).map(last_bd)),
                'CERN': pd.Series([311937692, 304348600, 305381551, 306589898, 301317068, 294222760, 294098094],
                                  index=pd.DatetimeIndex(['2020-01-28', '2020-04-23', '2020-07-22', '2020-10-21',
                                                          '2021-04-30', '2021-10-25', '2022-04-26']).map(last_bd)),
                'CTXS': pd.Series([123450644, 123123572, 124167045, 124230000, 124722872, 126579926, 126885081],
                                  index=pd.DatetimeIndex(['2020-04-28', '2020-10-23', '2021-04-29', '2021-06-30',
                                                          '2021-11-01', '2022-04-27', '2022-07-18']).map(last_bd)),
                'CXO': pd.Series([201028695, 196706121, 196701580, 196707339, 196304640],
                                 index=pd.DatetimeIndex(['2019-10-28', '2020-02-14', '2020-04-27',
                                                         '2020-07-26', '2020-10-23']).map(last_bd)),
                'DISCA': pd.Series([158566403, 160019717, 160205701, 160318208, 162490752, 169207249, 169580151],
                                   index=pd.DatetimeIndex(['2020-02-13', '2020-04-22', '2020-07-24', '2020-10-26',
                                                           '2021-02-08', '2021-10-22', '2022-02-10']).map(last_bd)),
                'DISCK': pd.Series([355843540, 340161506, 340170764, 324172931, 318331065, 330146263, 330153753],
                                   index=pd.DatetimeIndex(['2020-02-13', '2020-04-22', '2020-07-24', '2020-10-26',
                                                           '2021-02-08', '2021-10-22', '2022-02-10']).map(last_bd)),
                'DISH': pd.Series([284612148, 285722326, 286869973, 287530751, 287734942, 288909818, 289454037,
                                   290357069, 290571195, 291559614, 291869693, 292270989, 292716859, 294172528],
                                  index=pd.DatetimeIndex(['2020-02-10', '2020-04-27', '2020-07-31', '2020-10-22',
                                                          '2021-02-10', '2021-04-19', '2021-07-26', '2021-10-25',
                                                          '2022-02-14', '2022-04-25', '2021-07-22', '2022-10-24',
                                                          '2023-02-14', '2023-04-25']).map(last_bd)),
                'DRE': pd.Series([368382161, 370561785, 371951171, 374985270, 378340411, 380850300, 382767539,
                                  384455127, 384992716],
                                 index=pd.DatetimeIndex(['2020-04-29', '2020-07-29', '2020-10-27', '2021-04-28',
                                                         '2021-07-28', '2021-10-27', '2022-02-16', '2022-04-27',
                                                         '2022-08-04']).map(last_bd)),
                'EL': pd.Series([222319332, 224763197, 225569212, 226538215, 229736467, 231894845, 233045213],
                                index=pd.DatetimeIndex(['2020-01-30', '2020-04-24', '2020-08-20', '2020-10-26',
                                                        '2021-01-29', '2021-04-26', '2021-08-20']).map(last_bd)),
                'ETFC': pd.Series([221750841, 221046419, 221096380],
                                  index=pd.DatetimeIndex(['2020-02-14', '2020-04-30', '2020-08-03']).map(last_bd)),
                'FLIR': pd.Series([134455332, 130842358, 131121965, 131144505, 131238445, 131932461],
                                  index=pd.DatetimeIndex(['2020-02-25', '2020-05-01', '2020-07-31', '2020-10-23',
                                                          '2021-02-19', '2021-04-30']).map(last_bd)),
                'GOOG': pd.Series([340979832, 336162278, 333631113, 329867212, 327556472, 323580001, 320168491,
                                   317737778, 315639479, 313376417],
                                  index=pd.DatetimeIndex(['2020-01-27', '2020-04-21', '2020-07-23', '2020-10-22',
                                                          '2021-01-26', '2021-04-20', '2021-07-20', '2021-10-19',
                                                          '2022-01-25', '2022-04-19']).map(last_bd)),
                'GOOGL': pd.Series([299895185, 300050444, 300471156, 300643829, 300737081, 300746844, 301084627,
                                    300809676, 300754904, 300763622],
                                   index=pd.DatetimeIndex(['2020-01-27', '2020-04-21', '2020-07-23', '2020-10-22',
                                                           '2021-01-26', '2021-04-20', '2021-07-20', '2021-10-19',
                                                           '2022-01-25', '2022-04-19']).map(last_bd)),
                'INFO': pd.Series([392948672, 398916408, 396809671, 398358566, 398612292, 398841378, 399080370],
                                  index=pd.DatetimeIndex(['2019-12-31', '2020-02-29', '2020-05-31', '2020-08-31',
                                                          '2021-05-31', '2021-08-31', '2021-12-31']).map(last_bd)),
                'KSU': pd.Series([90964664, 90980440],
                                 index=pd.DatetimeIndex(['2021-07-09', '2021-10-12']).map(last_bd)),
                'MXIM': pd.Series([266625382, 266695209, 267301195, 268363654, 268566248],
                                  index=pd.DatetimeIndex(['2020-04-17', '2020-08-10', '2020-10-15',
                                                          '2021-04-16', '2021-08-10']).map(last_bd)),
                'NBL': pd.Series([479698676, 479768764],
                                 index=pd.DatetimeIndex(['2020-03-31', '2020-06-30']).map(last_bd)),
                'NLSN': pd.Series([356475591, 359941875],
                                  index=pd.DatetimeIndex(['2020-03-31', '2022-09-30']).map(last_bd)),
                'PBCT': pd.Series([433739103, 424657609, 424777066, 428020009],
                                  index=pd.DatetimeIndex(['2020-02-14', '2020-04-30', '2020-07-31',
                                                          '2021-10-31']).map(last_bd)),
                'PXD': pd.Series([165714771, 164863215, 164276170, 164406947, 216580280, 243952401, 243959045,
                                  244133701, 242884015, 241958985, 238666954, 237598733, 235004153, 233735537,
                                  233141153, 233308884, 233623121, 233675158],
                                 index=pd.DatetimeIndex(['2020-02-18', '2020-05-07', '2020-08-04', '2020-11-04',
                                                         '2021-02-22', '2021-05-06', '2021-08-04', '2021-11-03',
                                                         '2022-02-18', '2022-05-04', '2022-07-29', '2022-10-27',
                                                         '2023-02-21', '2023-04-25', '2023-07-31', '2023-11-01',
                                                         '2024-02-20', '2024-05-01']).map(last_bd)),
                'RTN': pd.Series([278479000, 278441000],
                                 index=pd.DatetimeIndex(['2019-10-21', '2020-02-10']).map(last_bd)),
                'SBNY': pd.Series([60632000, 63065000, 62929000, 62927000, 62929000, 62250000],
                                  index=pd.DatetimeIndex(['2021-11-15', '2022-02-15', '2022-05-15', '2022-08-15',
                                                          '2022-11-15', '2023-01-31']).map(last_bd)),
                'SIVB': pd.Series([51513227, 51796902, 54315140, 56436504, 58687392, 58802627, 58851167, 59082305,
                                   59104124, 59200925],
                                  index=pd.DatetimeIndex(['2020-04-30', '2020-10-31', '2021-04-30', '2021-07-31',
                                                          '2021-10-31', '2022-01-31', '2022-04-30', '2022-07-31',
                                                          '2022-10-31', '2023-01-31']).map(last_bd)),
                'TIF': pd.Series([121346674, 121368585, 121411166],
                                 index=pd.DatetimeIndex(['2020-04-30', '2020-07-31', '2020-10-31']).map(last_bd)),
                'TWTR': pd.Series([782287089, 784629121, 790948853, 795349591, 798152488, 798126631, 799609869,
                                   800641166, 764180688, 765246152],
                                  index=pd.DatetimeIndex(['2020-02-06', '2020-04-30', '2020-07-23', '2020-10-29',
                                                          '2021-02-09', '2021-04-23', '2021-10-22',
                                                          '2022-02-10', '2022-04-22', '2022-07-22']).map(last_bd)),
                'V': pd.Series([1706024403, 1687112437, 1686007156, 1692383762, 1696113603, 1691806129, 1687643027,
                                1669730762, 1658423632, 1645719350, 1635014650, 1628169181, 1624954064, 1618223392],
                               index=pd.DatetimeIndex(['2020-01-24', '2020-04-30', '2020-07-24', '2020-11-13',
                                                       '2021-01-22', '2021-04-23', '2021-07-23', '2021-11-10',
                                                       '2022-01-29', '2022-04-20', '2022-07-20', '2022-11-09',
                                                       '2023-01-18', '2023-04-19']).map(last_bd)),
                'VAR': pd.Series([90814945, 90941138, 91355469, 91838813],
                                 index=pd.DatetimeIndex(['2020-05-01', '2020-07-31', '2020-11-13',
                                                         '2021-01-29']).map(last_bd)),
                'WCG': pd.Series([50312077, 50327612],
                                 index=pd.DatetimeIndex(['2019-07-26', '2019-10-28']).map(last_bd)),
                'WRK': pd.Series([258456273, 259255002, 259636357, 262653756, 263516869, 266116343, 267006103,
                                  265001543, 263214392, 254851968, 254298051, 254463987, 254651783, 256130237,
                                  256279376, 256469100, 256966731, 258148056],
                                 index=pd.DatetimeIndex(['2020-01-17', '2020-04-24', '2020-07-24', '2020-11-06',
                                                         '2021-01-22', '2021-04-23', '2021-07-23', '2021-11-05',
                                                         '2022-01-21', '2022-05-03', '2022-07-22', '2022-11-04',
                                                         '2023-01-20', '2023-04-21', '2023-07-21', '2023-11-03',
                                                         '2024-01-19', '2024-04-19']).map(last_bd)),
                'XEC': pd.Series([101810140, 102135577],
                                 index=pd.DatetimeIndex(['2019-10-31', '2020-01-31']).map(last_bd)),
                'XLNX': pd.Series([248836561, 243846000, 244314000, 245059000, 245277000, 245840000, 247468170,
                                   247880415, 248382008],
                                  index=pd.DatetimeIndex(['2020-01-10', '2020-04-24', '2020-07-10', '2020-10-09',
                                                          '2021-01-15', '2021-04-30', '2021-07-16', '2021-10-15',
                                                          '2022-01-14']).map(last_bd))}


class EuropeBanksStockMarketMetrics(Metrics):
    def __init__(self, tickers, additional_share_classes=None, stock_index=None, start=None, hist_shares_outs=None,
                 currency_conversion_df=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a dictionary representing all the ticker symbols making up a stock market that this class
                        derives metrics for. Each key represents a ticker symbol, each value the start and end day of
                        the stock represented by the ticker being added or removed. An end value of None implies it's
                        still part of the market, a start value of None designates it becoming part of the market
                        before 'start'
        :param additional_share_classes: a dictionary whose keys are additional share classes of companies
                                         that have multiple share classes and where they all are part of the market,
                                         the values are the first share class (typically class A)
        :param stock_index: an optional ticker symbol of the index that corresponds to the market represented
                            by the 'tickers' parameter
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many kinds of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param hist_shares_outs: a dictionary representing historical shares outstanding for delisted tickers.
                            Each key represents a ticker symbol. Each value is a panda Series designating shares
                            outstanding on certain days.
        :param currency_conversion_df: for heterogeneous markets made up of stocks priced in different currencies,
                               a DataFrame whose columns are suffixes such as '.L', '.SW', '.CO', etc. and whose
                               rows are the corresponding conversion rates.
        """
        # Unfortunately Yahoo-Finance reports incorrect closing prices for shares before they had a stock split
        super().__init__(tickers, additional_share_classes, stock_index, start, hist_shares_outs,
                         None, currency_conversion_df)

        import eurostat
        # Using 1-Year spot rate of Eurozone AAA-rated government bonds as proxy for riskless rate
        euro_curves = eurostat.get_data_df('irt_euryld_d', filter_pars={
            'startPeriod': (start - pd.DateOffset(months=1)).date(), 'freq': 'D',
            'yld_curv': 'SPOT_RT',
            'maturity': 'Y1',
            'bonds': 'CGB_EA_AAA', 'geo': 'EA'})
        euro_curves = euro_curves.drop(euro_curves.columns[:2].append(euro_curves.columns[3:5]), axis=1)
        euro_curves = euro_curves.set_index(euro_curves.columns[0]).T / 100.
        euro_curves = euro_curves.set_axis(pd.DatetimeIndex(euro_curves.index))

        # Convert from continuous to annual compounding
        self.riskless_rate = np.exp(euro_curves.Y1.astype('float64')) - 1.

    @staticmethod
    def get_stoxx_europe_banks_components():
        """
        Returns a list of ticker symbols of Stoxx Europe 600 Banks Index.
        """
        return ['INGA.AS', 'ABN.AS', 'DBK.DE', 'CBK.DE', 'BNP.PA', 'ACA.PA', 'GLE.PA', 'KBC.BR', 'BIRG.IR',
                'BBVA.MC', 'BKT.MC', 'CABK.MC', 'SAB.MC', 'SAN.MC', 'EBS.VI', 'RBI.VI', 'NDA-FI.HE',
                'ISP.MI', 'UCG.MI', 'MB.MI', 'BAMI.MI', 'BPE.MI',
                'HSBA.L', 'BARC.L', 'LLOY.L', 'NWG.L', 'VMUK.L', 'STAN.L',
                'BAER.SW', 'UBSG.SW', 'CMBN.SW',
                'SEB-A.ST', 'SWED-A.ST', 'SHB-A.ST',
                'DANSKE.CO', 'SYDB.CO', 'JYSK.CO',
                'DNB.OL',
                'KOMB.PR']

    @staticmethod
    def tickers_to_dict(tickers, st):
        """
        Converts a list of ticker symbols into a dictionary whose keys are the same ticker symbols and whose values
        are (start-date, end-date) pairs indicating since when a particular ticker has been part of the market
        """
        return {ticker: (st, None) for ticker in tickers}