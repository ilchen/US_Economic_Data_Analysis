import numpy as np
import pandas as pd
from pandas.tseries.offsets import YearBegin, BYearEnd, BDay, BMonthEnd, MonthBegin, QuarterEnd
import pandas_datareader.data as web
from math import sqrt, isclose
from collections import defaultdict
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
    INDEX_TO_VOLATILITY_MAP = {
        '^GSPC': '^VIX'
    }
    UNIDENTIFIED_SECTOR = 'unidentified'
    BANKING_INDUSTRIES = {'capital-markets', 'banks-diversified', 'banks-regional'}

    def __init__(self, tickers, additional_share_classes=None, stock_index=None, start=None, hist_shares_outs=None,
                 tickers_to_correct_for_splits=None, currency_conversion_df=None):
        """
        Constructs a Metrics object out of a list of ticker symbols and an optional index ticker

        :param tickers: a dictionary representing all the ticker symbols making up a stock market that this class
                        derives metrics for. Each key represents a ticker symbol. Each value designates a list of pairs,
                        each pair representing the dates of inclusion and exclusion from the index during a specific
                        period (the inclusion period includes the exclusion date). A start date of 'None' in a pair
                        means the ticker was part of the market before 'start'. An end date of 'None' means
                        the ticker is still part of the market.
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
        # self.data.index = self.data.index.tz_localize(None)

        # If some stocks that are no longer part of the index and are no longer listed were traded over the counter,
        # while stocks still in the index were not traded, we need to make an adjustment
        idx = self.data.loc[:, (Metrics.CLOSE, self.get_current_components())].dropna(how='all').index
        if len(idx) < len(self.data):
            self.data = self.data.loc[idx]

        # For markets composed of stocks traded on different exchanges such as SX7V, it can happen that some stocks
        # were not traded on the last trading day. In this case forward-fill their prices and trading volumes
        # from the last trading day
        col_idx = self.data.isna().iloc[-1]
        if len(self.data.loc[self.data.index[-1], ~col_idx]) > len(self.data.loc[self.data.index[-1], col_idx]):
            self.data.loc[self.data.index[-1], col_idx] = self.data.loc[self.data.index[-2], col_idx]

        # In case some stocks captured in the tickers list were not trading during the date range,
        # I assume they were trading using their first price.
        self.data.bfill(inplace=True)

        # Unfortunately Yahoo-Finance provides dividend yield only for the most recent trading day
        self.dividend_yield = {}
        self.pe = {}
        self.eps = {}
        for ticker in self.get_current_components():
            info = self.tickers.tickers[ticker].info
            dividend_yield = info.get('dividendYield')
            pe = info.get('forwardPE')
            eps = info.get('forwardEps')

            # A kludge for London Stock Exchange, Yahoo-Finance sometimes reports incorrect forwardPEs
            sfx = self.get_exchange_suffix(ticker)
            if sfx == '.L':
                trailing_pe = info.get('trailingPE')
                if trailing_pe is not None:
                    if trailing_pe / pe < .05:
                        trailing_eps = info.get('trailingEps')
                        if trailing_eps is not None and eps is not None:
                            pe = trailing_pe * trailing_eps / eps
                    elif pe / trailing_pe < .05:
                        pe *= 100.

            self.dividend_yield[ticker] = 0. if dividend_yield is None else dividend_yield
            self.pe[ticker] = 0. if pe is None else pe
            self.eps[ticker] = 0. if eps is None else eps

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
        for delisted_ticker in [k for (k, periods) in tickers.items() if periods[-1][1] is not None]:
            ed = min(tickers[delisted_ticker][-1][1], self.data.index[-1])
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
            cur_comps_set = set(self.get_current_components())
            for ticker in self.ticker_symbols.keys():
                sfx = self.get_exchange_suffix(ticker)
                if sfx in currency_conversion_df.columns:
                    self.data.loc[:, (self.CLOSE, ticker)] *= currency_conversion_df.loc[self.data.index, sfx]
                    if ticker in cur_comps_set:
                        self.eps[ticker] *= currency_conversion_df.loc[self.data.index[-1], sfx]

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
                    if not shares_outstanding2:
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

                    # As of late Yahoo-Finance reports correct shares outstanding data for some delisted shares.
                    # I use these data from Yahoo-Finance in preference to more coarse-grained data from overrides
                    # when it is clear that it is more accurate
                    overlap_idx = shares_outst.index.intersection(correction.index)
                    if len(overlap_idx) < 5  or  ((shares_outst.loc[overlap_idx] - correction[overlap_idx])
                                                   / shares_outst.loc[overlap_idx]).abs().mean() > 0.02:
                        shares_outst.update(correction)
                        print('\tExtra correction for the number of shares outstanding for {:s} '
                              'from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(ticker, self.data.index[0], correction.index[-1]))

            # Yahoo-Finance doesn't report shares outstanding for each trading day. I compensate for it by rolling
            # forward the most recent reported value and then rolling backward the earliest available number.
            missing_dates = self.data.index.difference(shares_outst.index)
            shares_outst = pd.concat([shares_outst, pd.Series(np.nan, index=missing_dates)]).sort_index()
            shares_outst = shares_outst.ffill().bfill()

            # Getting rid of extraneous dates
            self.shares_outstanding[ticker] = shares_outst.loc[self.data.index]

        num_countries = len({self.get_exchange_suffix(ticker) for ticker in self.get_current_components()})
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

            for (st, ed) in self.ticker_symbols[ticker]:
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

                    # Calculating forward P/E for the whole market using a capitalization-weighted approach
                    self.forward_PE += self.shares_outstanding[ticker].iloc[-1] * self.eps[ticker]

        self.forward_dividend_yield /= self.capitalization.iloc[-1,0]
        self.forward_PE = self.capitalization.iloc[-1,0] / self.forward_PE

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

    @staticmethod
    def validate_method(method):
        allowed_methods = {'mean', 'first', 'last', 'min', 'max'}
        if method not in allowed_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from: {', '.join(allowed_methods)}")

    def validate_tickers_in_market(self, tickers):
        if tickers is None or not set(tickers) <= self.ticker_symbols.keys():
            raise ValueError("Ticker symbols passed don't represent a subset of the market. "
                             'The following ticker symbols are not part of the market: '
                             f'{set(tickers) - self.ticker_symbols.keys()}')

    def get_capitalization(self, frequency='ME', method='mean', tickers=None):
        """
        Calculates the capitalization of a given market or a subset of stocks over time.
        If a frequency less granular than daily is provided, the data is downsampled accordingly.
        For the 'mean' method, the result is the average capitalization over each period.
        For other supported methods ('first', 'last', 'min', 'max'), the capitalization is derived
        from the respective day or value within each period.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param method: aggregation strategy applied during resampling, one of 'mean', 'first', 'last', 'min', 'max'
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the capitalization of the market
        """
        self.validate_method(method)
        if tickers is None:
            return self.capitalization.loc[:, Metrics.CAPITALIZATION].resample(frequency).agg(method).dropna()
        else:
            ret = pd.Series(0., index=self.data.index)
            for ticker in tickers:
                for (st, ed) in self.ticker_symbols[ticker]:
                    ret += self.data.loc[st:ed, (Metrics.CLOSE, ticker)] * self.shares_outstanding[ticker].loc[st:ed]
            return ret if frequency in ['B', 'D', 'C'] else ret.resample(frequency).agg(method).dropna()

    def adjust_for_additional_share_classes(self, cap_series):
        for additional_share_class, main_share_class in self.additional_share_classes.items():
            cap_series.loc[main_share_class] += cap_series.loc[additional_share_class]
            cap_series.drop(additional_share_class, inplace=True)

    def get_capitalization_for_companies(self, tickers, frequency='ME', method='mean',
                                         merge_additional_share_classes=True):
        """
        Calculates the capitalization of stocks represented by the 'tickers' parameter.
        If a frequency less granular than daily is provided, the data is downsampled accordingly.
        For the 'mean' method, the result is the average capitalization over each period.
        For other supported methods ('first', 'last', 'min', 'max'), the capitalization is derived
        from the respective day or value within each period.

        :param tickers: a list of one or more ticker symbols
        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param method: aggregation strategy applied during resampling, one of 'mean', 'first', 'last', 'min', 'max'
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.DataFrame object capturing the capitalization of the companies represented by the 'tickers' parameter
        """
        self.validate_tickers_in_market(tickers)
        self.validate_method(method)
        cap_df = self.capitalization.loc[:, [self.CAPITALIZATION] + tickers]
        if merge_additional_share_classes and len(self.additional_share_classes) > 0\
                and set(tickers) & self.additional_share_classes.keys():
            cap_df = cap_df.copy()
            for additional_share_class, main_share_class in self.additional_share_classes.items():
                if additional_share_class in tickers and main_share_class in tickers:
                    cap_df.loc[:, main_share_class] += cap_df.loc[:, additional_share_class]
                    cap_df.drop(additional_share_class, axis=1, inplace=True)
        return cap_df if frequency in ['B', 'D', 'C'] else cap_df.resample(frequency).agg(method).dropna()

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

    def get_top_n_capitalization_companies_for_month(self, n, dt=None, method='mean',
                                                     merge_additional_share_classes=True):
        """
        Calculates the top-n capitalization companies for the market taking the average for the month specified.

        :param n: the number of maximum capitalization companies to return
        :param dt: (string, int, date, datetime, Timestamp) – the date for whose month to return the topn capitalization
                    if None, the most recent business day is used
        :param method: aggregation strategy applied during resampling, one of 'mean', 'first', 'last', 'min', 'max'
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.Series object capturing the top-n capitalization companies for the market along with the
                  percentage share of their capitalization in the market. The pd.Series is indexed by the ticker
                  symbols.
        """
        self.validate_method(method)
        if dt is None:
            dt = self.data.index[-1]
        dt = MonthBegin(0).rollback(dt) if MonthBegin(0).rollback(dt) <= self.data.index[-1]\
            else MonthBegin(0).rollback(self.data.index[-1])
        resampled_cap = self.capitalization.resample('MS').agg(method)
        cap_series = resampled_cap.loc[dt, resampled_cap.columns[2:]]
        if merge_additional_share_classes and len(self.additional_share_classes) > 0:
            self.adjust_for_additional_share_classes(cap_series)
        topn = cap_series.nlargest(n).to_frame(self.CAPITALIZATION)
        topn[self.MKT_SHARE] = topn.iloc[:, 0] / resampled_cap.loc[dt, self.CAPITALIZATION]
        return topn

    def get_top_n_capitalization_companies_for_year(self, n, dt=None, method='mean',
                                                    merge_additional_share_classes=True):
        """
        Calculates the top-n capitalization companies for the market taking the average for the year specified.

        :param n: the number of maximum capitalization companies to return
        :param dt: (string, int, date, datetime, Timestamp) – the date for whose year to return the topn capitalization
                    if None, the current year is used
        :param method: aggregation strategy applied during resampling, one of 'mean', 'first', 'last', 'min', 'max'
        :param merge_additional_share_classes: for companies that have multiple share classes listed (e.g. Google),
                                               indicates whether to merge the capitalization of different share classes
                                               into one
        :returns: a pd.Series object capturing the top-n capitalization companies for the market along with the
                  percentage share of their capitalization in the market. The pd.Series is indexed by the ticker
                  symbols.
        """
        self.validate_method(method)
        if dt is None:
            dt = self.data.index[-1]
        dt = YearBegin(0).rollback(dt) if YearBegin(0).rollback(dt) <= self.data.index[-1]\
            else YearBegin(0).rollback(self.data.index[-1])
        resampled_cap = self.capitalization.resample('YS').agg(method)
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
                for (st, ed) in self.ticker_symbols[ticker]:
                    daily_turnover += self.data.loc[st:ed, (Metrics.CLOSE, ticker)]\
                        * self.data.loc[st:ed, (Metrics.VOLUME, ticker)]
        return daily_turnover

    def get_daily_trading_value(self, frequency='ME', tickers=None):
        """
        Calculates a total daily trading value of stocks in the market implied by this object.
        Downsamples if a less granular frequency than daily is specified. Takes a cumulative turnover over periods
        implied by the 'frequency' parameter and then annualizes it.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the turnover
        """
        return self.get_daily_trading_value_ds(tickers).resample(frequency).mean().dropna().rename(self.TOTAL_VALUE)

    def get_annual_trading_value(self, frequency='ME', tickers=None):
        """
        Calculates the annual trading value of stocks based on resampled daily trading data.

        The method resamples daily trading values to the specified frequency, computing an annualized total.
        For monthly-like frequencies ('ME', 'MS', 'BMS', 'BME'), it sums the monthly averages and multiplies by 12.
        For year-end frequencies ('YS', 'YE', 'BYS', 'BYE'), it directly sums the yearly values.
        If the final period does not cover a complete month or year, the method estimates the last value
        using the most recent average trading activity.

        :param frequency: Pandas offset alias indicating the desired resampling frequency
            (e.g., 'ME' for month-end, 'YE' for year-end).
            See: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param tickers: Optional list of ticker symbols to filter daily trading value calculations.
        :returns: pd.Series of annualized trading value per period at the specified frequency.
        """
        tmp = self.get_daily_trading_value_ds(tickers)
        ret = tmp.resample(frequency).mean().dropna().rename(self.TOTAL_VALUE) * self.TRADING_DAYS_IN_YEAR
        if frequency in ['ME', 'MS', 'BMS', 'BME']:
            ret2 = tmp.resample(frequency).sum().rename(self.TOTAL_VALUE) * 12
            # In case it's not a full month, use an estimate from daily value traded
            if tmp.index[-1] < tmp.index[-1] + BMonthEnd(0):
                ret2.iloc[-1] = ret.iloc[-1]
            return ret2
        elif frequency in ['YS', 'YE', 'BYS', 'BYE']:
            ret2 = tmp.resample(frequency).sum().rename(self.TOTAL_VALUE)
            # In case it's not a full year, use an estimate from daily value traded
            if tmp.index[-1] < tmp.index[-1] + BYearEnd(0):
                ret2.iloc[-1] = ret.iloc[-1]
            return ret2
        return ret

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
        daily_turnover /= self.get_capitalization('B', tickers=tickers) if tickers\
            else self.capitalization.loc[:, Metrics.CAPITALIZATION]
        return daily_turnover.resample(frequency).mean().dropna()

    def get_annual_turnover(self, frequency='ME', method='mean', tickers=None):
        """
        Calculates an annual turnover ratio of stocks in the market implied by this object to its capitalization.
        If a frequency less granular than daily is specified, data is downsampled accordingly.
        The annual trading value is divided by capitalization resampled to the target frequency and
        aggregated using the specified method.

        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param method: aggregation strategy applied during resampling, one of 'mean', 'first', 'last', 'min', 'max'
        :param tickers: a list of one or more ticker symbols
        :returns: a pd.Series object capturing the turnover
        """
        self.validate_method(method)
        if frequency in ['ME', 'MS', 'BMS', 'BME', 'YS', 'YE', 'BYS', 'BYE']:
            ret = self.get_annual_trading_value(frequency, tickers)
            if tickers is None:
                ret /= self.capitalization.loc[:, Metrics.CAPITALIZATION].resample(frequency).agg(method)
            else:
                ret /= self.get_capitalization('B', tickers=tickers).resample(frequency).agg(method)
            return ret
        return self.get_daily_turnover(frequency, tickers) * self.TRADING_DAYS_IN_YEAR

    def get_annual_volatility(self, alpha=1-.94453, frequency='ME', include_next_month=False):
        """
        Calculates an annual volatility of a market represented by the stock market index used when constructing
        this instance of Metrics. It uses an exponentially weighted moving average (EWMA) with a given smoothing factor
        alpha. It downsamples it if a less granular frequency than daily is specified. When downsampling it takes
        an average volatility over periods implied by the 'frequency' parameter.

        :param alpha: a smoothing factor alpha. Then calculating the EWMA, the most recent observation will be
                      multiplied by 'alpha', while the previous estimate of EMWA by '(1-alpha)'
        :param frequency: a standard Pandas frequency designator
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param include_next_month: indicates if the implied volatility for the next month derived from index options
                                   should be included
        :returns: a pd.Series object capturing the volatility
        """
        if self.stock_index_data is None:
            return None
        vol = self.stock_index_data.pct_change().ewm(alpha=alpha).std() * self.TO_ANNUAL_MULTIPLIER
        st = MonthBegin(0).rollback(self.data.index[0])
        ret = vol.resample(frequency).mean().dropna().rename(self.VOLATILITY).loc[st:]
        idx_name = self.stock_index_data.name[1] if self.stock_index_data.name is not None else None
        if include_next_month and idx_name is not None and idx_name in self.INDEX_TO_VOLATILITY_MAP\
                and frequency in ['ME', 'MS']:
            impl_volatility = yfin.Ticker(self.INDEX_TO_VOLATILITY_MAP[idx_name]).history(period='5d')
            impl_volatility = impl_volatility.loc[:, self.CLOSE].resample(frequency).last() / 100.
            impl_volatility.index = impl_volatility.index.shift(1)
            ret = pd.concat([ret, impl_volatility.tz_localize(None).iloc[-1:]])
        return ret

    def get_current_components(self):
        """
        Returns the current set of components in a given stock market.
        """
        return [k for k, periods in self.ticker_symbols.items() if periods[-1][1] is None]

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
            # and not close prices
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

    def get_banking_sector_components(self):
        """
        Returns a list of ticker symbols from the market this object represents that belong to the banking sector.
        The banking sector is defined as companies belonging to the 'Financial Services' sector and one of the following
        three industries: banks-regional, banks-diversified, capital-markets. The latter represents banks such as
        The Goldman Sachs and Morgan Stanley.
        """
        ret = []
        for ticker in self.get_current_components():
            if ticker in self.additional_share_classes:
                # Ignoring additional share class for {ticker} to avoid double counting'
                continue
            if self.tickers.tickers[ticker].info.get('industryKey') in self.BANKING_INDUSTRIES:
                ret.append(ticker)
        return ret

    def get_sector_breakdown_for_day(self, dt=None):
        """
        Compute the sector breakdown for the market on a specified day, based on relative capitalization.
        
        Parameters:
            dt (str | int | date | datetime | pd.Timestamp, optional): 
                The date for which to compute the sector breakdown. 
                If None, defaults to the most recent business day available.
        
        Returns:
            pd.Series: A series representing the market breakdown by sector, 
                       with values proportional to each sector's total market capitalization.
        """
        if dt is None:
            dt = self.data.index[-1]
        idx = -1
        dt = BDay(0).rollback(dt) if BDay(0).rollback(dt) <= self.data.index[idx] else self.data.index[idx]
        while np.isnan(self.capitalization.loc[dt, self.CAPITALIZATION]):
            idx -= 1
            dt = self.data.index[idx]

        ret = defaultdict(float)

        for ticker in self.ticker_symbols.keys():
            if not self.is_ticker_in_market(ticker, dt): continue
            sector_key = self.tickers.tickers[ticker].info.get('sectorKey', Metrics.UNIDENTIFIED_SECTOR)
            ret[sector_key] += self.data.loc[dt, (Metrics.CLOSE, ticker)]\
                                  * self.shares_outstanding[ticker].loc[dt]

        return pd.Series(ret) / self.capitalization.loc[dt, Metrics.CAPITALIZATION]

    def tickers_to_display_names(self, tickers, max_len=15):
        """
        Returns displayable names for ticker symbols passed via the 'tickers' parameter.

        :param tickers: an iterable object containing ticker symbols whose display names need to be looked up
        :param max_len: designates the maximum length of a name, in case a name is longer than 'max_len' it gets
                        trimmed to 'max_len' with the last two symbols being '..', it must be at least 7 symbols long.
                        If you want a shorter name, just use the ticker symbol.
        :returns: a list object having the same length as the 'tickers' parameter and containing the corresponding
                  displayable names for each ticker
        """
        if max_len < 7:
            raise ValueError(f"max_len of {max_len} is too short, you'd better use ticker symbols directly")
        ret = []
        market_comps = self.ticker_symbols.keys()
        for ticker in tickers:
            if ticker in market_comps and self.tickers.tickers[ticker]:
                dn = self.tickers.tickers[ticker].info.get('displayName')\
                      or self.tickers.tickers[ticker].info.get('shortName')\
                      or self.tickers.tickers[ticker].info.get('longBusinessSummary')
                if dn.lower().startswith('the '):
                    dn = dn[4:]
                if len(dn) > max_len:
                    dn = dn[:max_len].rstrip()
                    dn = dn[:len(dn)-2] + '..'
                ret.append(dn)
        return ret

    @staticmethod
    def get_exchange_suffix(ticker):
        comps = ticker.split('.')
        sfx = '.' + comps[-1]
        return None if len(comps) == 1 else sfx

    @staticmethod
    def tickers_to_dict(tickers, st):
        """
        Converts a list of ticker symbols into a dictionary whose keys are the same ticker symbols and whose values
        are lists with one (start-date, end-date) pair indicating since when a particular ticker has been part of
        the market
        """
        return {ticker: [(st, None)] for ticker in tickers}

    def is_ticker_in_market(self, ticker, dt):
        """
        Check if the given ticker is part of the market on the specified date.

        :param ticker: str, the ticker symbol to check
        :param dt: str, datetime, or Timestamp, the date to check
        :return: bool, True if the ticker is in the market on the date, False otherwise
        """
        # Convert the input date to a pandas Timestamp
        dt = pd.to_datetime(dt)

        # Check each period
        for entry, exit in self.ticker_symbols[ticker]:
            if entry <= dt and (exit is None or dt <= exit):
                return True

        # No matching period found
        return False

    @staticmethod
    def get_historical_components(cur_components, file_name, start=None):
        """
        Returns a dictionary whose keys are ticker symbols representing companies that were part of a market
        at any time since 'start' and whose values are lists of pairs, each pair representing the dates of
        inclusion and exclusion from the index during a specific period (the inclusion period includes
        the exclusion date). A start date of 'None' in a pair
        means the ticker was part of the index before 'start'. An end date of 'None' means the ticker is
        still part of the index.

        :param cur_components: list of current ticker symbols in the market.
        :param file_name: str, path to the CSV file with historical changes.
        :param start: an int, float, str, datetime, or date object designating the starting time.
        """
        from datetime import date
        if type(start) is date:
            start = datetime.combine(start, time())
        start = pd.to_datetime(start)
        if pd.isna(start):
            start = pd.to_datetime('1900-01-01')  # Use an early date if start is None

        # Read and filter CSV data
        df = pd.read_csv(file_name, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df[df.index >= start].sort_index()  # Sort chronologically

        # Collect all tickers involved
        all_tickers = set(cur_components)
        for _, row in df.iterrows():
            if not pd.isnull(row.iloc[0]):
                all_tickers.update(row.iloc[0].split(','))
            if not pd.isnull(row.iloc[1]):
                all_tickers.update(row.iloc[1].split(','))

        # Initialize periods dictionary
        periods = {ticker: [] for ticker in all_tickers}

        # Collect events for each ticker
        events = {ticker: [] for ticker in all_tickers}
        for date, row in df.iterrows():
            if not pd.isnull(row.iloc[0]):  # Process additions
                for ticker in row.iloc[0].split(','):
                    if ticker in all_tickers:
                        events[ticker].append((date, 'add'))
            if not pd.isnull(row.iloc[1]):  # Process removals
                for ticker in row.iloc[1].split(','):
                    if ticker in all_tickers:
                        events[ticker].append((date, 'remove'))

        # Sort events chronologically for each ticker
        for ticker in events:
            events[ticker].sort(key=lambda x: x[0])

        # Build periods for each ticker
        for ticker in all_tickers:
            ticker_events = events[ticker]
            if not ticker_events:  # No events since start
                if ticker in cur_components:
                    periods[ticker].append((start, None))
                continue

            # If first event is 'remove', ticker was in index at start
            if ticker_events[0][1] == 'remove':
                remove_date = ticker_events[0][0]
                periods[ticker].append((start, remove_date - BDay(1)))
                start_idx = 1
            else:
                start_idx = 0

            # Process subsequent events in pairs
            i = start_idx
            while i < len(ticker_events):
                if ticker_events[i][1] == 'add':
                    add_date = ticker_events[i][0]
                    i += 1
                    if i < len(ticker_events) and ticker_events[i][1] == 'remove':
                        remove_date = ticker_events[i][0]
                        periods[ticker].append((add_date, remove_date - BDay(1)))
                        i += 1
                    else:
                        # No subsequent 'remove'
                        if ticker in cur_components:
                            periods[ticker].append((add_date, None))
                        else:
                            raise ValueError(
                                f"Ticker {ticker} has an 'add' without a 'remove' but is not in cur_components")
                else:
                    # Unexpected 'remove' without preceding 'add'
                    i += 1  # Skip to next event

        return periods

    def get_quarterly_stmt_data(self, quarters=5, currency_conversion_df=None, tickers=None):
        """
        Calculates a cumulative value of a subset of quarterly income and cashflow statements' lines for all companies
        making up the market represented by this object.

        :param quarters: Number of trailing quarters to include in the aggregation
        :param currency_conversion_df: for heterogeneous markets made up of stocks priced in different currencies,
                       a DataFrame whose columns are suffixes such as '.L', '.SW', '.CO', etc. and whose
                       rows are the corresponding conversion rates.
        :param tickers: a list of one or more ticker symbols, if None the whole market implied by this object
                        is analyzed
        :returns: Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        - The first DataFrame contains market-wide cumulative values.
        - The second maps sector keys to their respective cumulative DataFrames.
        """
        if tickers:
            self.validate_tickers_in_market(tickers)
        last_qe = QuarterEnd(0).rollback(date.today())
        first_qe = QuarterEnd(0).rollback(max(self.data.index[0], last_qe - pd.DateOffset(months=(quarters-1)*3)))

        # Unfortunately 'Operating Income', 'Interest Expense', 'EBITDA', 'EBIT' are too frequently missing
        # => not worthwhile adding them
        # From cashflow statements only taking 'Capital Expenditure' and  'Cash Dividends Paid'
        idx = pd.Index(['Total Revenue', 'Pretax Income', 'Net Income', 'Tax Provision',
                        'Capital Expenditure', 'Purchase Of Business', 'Cash Dividends Paid'])
        idx_istmt = idx[0:4]
        df_factory = lambda: pd.DataFrame(0., index=idx, columns=pd.date_range(first_qe, last_qe, freq='QE')[::-1])
        result = df_factory()
        result_sectors = defaultdict(df_factory)
        for ticker in self.ticker_symbols.keys() if not tickers else tickers:
            if ticker in self.additional_share_classes:
                print(f'Ignoring additional share class for {ticker} to avoid double counting')
                continue

            # Only looking at the last period of being part of the market
            (st, ed) = self.ticker_symbols[ticker][-1]
            st = pd.Timestamp(st)

            # The company fell out of the market before the start of the timeframe under analysis
            if ed is not None and pd.Timestamp(ed) < first_qe:
                continue

            # To ensure we are able to compare 'ed' with Timestamps
            if ed is None:
                ed = last_qe
            else:
                ed = pd.Timestamp(ed)

            istmt = self.tickers.tickers[ticker].quarterly_income_stmt
            cfstmt = self.tickers.tickers[ticker].quarterly_cash_flow

            if len(istmt.columns) == 0:
                print(f"Yahoo Finance doesn't have income statement for {ticker} for the following quarters")
                print(result.columns[(result.columns >= st) & (result.columns <= ed)])
                continue

            if len(cfstmt.columns) == 0:
                print(f"Yahoo Finance doesn't have cashflow statement for {ticker} for the following quarters")
                print(result.columns[(result.columns >= st) & (result.columns <= ed)])

            # Currency conversion if required
            sfx = self.get_exchange_suffix(ticker)
            if currency_conversion_df is not None and sfx in currency_conversion_df.columns:
                convs = currency_conversion_df.loc[istmt.columns, sfx]
                istmt *= convs

                if len(cfstmt.columns) != 0:
                    convs = currency_conversion_df.loc[cfstmt.columns, sfx]
                    cfstmt *= convs

            # Make sure we align on calendar quarter end for those companies whose financial quarters
            # are not calendar-aligned
            istmt = istmt.rename(QuarterEnd(0).rollforward, axis=1)
            cfstmt = cfstmt.rename(QuarterEnd(0).rollforward, axis=1)

            # Remove duplicate columns due to data quality issues
            istmt = istmt.loc[:, ~istmt.columns.duplicated()]
            cfstmt = cfstmt.loc[:, ~cfstmt.columns.duplicated()]

            if istmt.columns[0] > last_qe and (result.columns[4] not in istmt.columns
                                               or (istmt.loc[idx_istmt, result.columns[4]]).isna().all()):
                print(f'Need to provide values for {ticker} for {result.columns[4]:%Y-%m-%d}')

            if istmt.columns[0] < last_qe <= ed:
                print(f'Need to provide values for {ticker} for {last_qe:%Y-%m-%d}')

            istmt = istmt.loc[:, last_qe:].astype('float64')
            cfstmt = cfstmt.loc[:, last_qe:].astype('float64')

            # Make sure that we trim to the end of the last calendar quarter
            common_quarters = result.columns.intersection(istmt.columns)
            if len(cfstmt.columns) != 0:
                common_quarters = common_quarters.intersection(cfstmt.columns)
            common_idx = istmt.index.intersection(idx)
            common_idx_cf = cfstmt.index.intersection(idx)

            # Adjusting for when the company's stock was part of the market
            common_quarters = common_quarters[(common_quarters >= st) & (common_quarters <= ed)]
            if len(common_quarters) == 0:
                # The stock wasn't part of the market during the timeframe under analysis
                continue

            if len(common_idx) < len(idx_istmt):
                print(f'For {ticker} missing the following income statement lines:')
                print(idx_istmt.difference(common_idx))
            if istmt.loc[common_idx, common_quarters].loc[:, common_quarters[0]].isna().any():
                print(f'Missing {common_quarters[0]:%Y-%m-%d} quarter data for {ticker}'
                      ' for the following income statement lines:')
                print(common_idx[istmt.loc[common_idx, common_quarters].loc[:, common_quarters[0]].isna()])

            sector_key = self.tickers.tickers[ticker].info.get('sectorKey', Metrics.UNIDENTIFIED_SECTOR)
            istmt_values = istmt.loc[common_idx, common_quarters].fillna(0.)
            result.loc[common_idx, common_quarters] += istmt_values
            result_sectors[sector_key].loc[common_idx, common_quarters] += istmt_values
            if len(cfstmt.columns) != 0:
                cfstmt_values = cfstmt.loc[common_idx_cf, common_quarters].fillna(0.)
                result.loc[common_idx_cf, common_quarters] += cfstmt_values
                result_sectors[sector_key].loc[common_idx_cf, common_quarters] += cfstmt_values

        return result, result_sectors


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
                          'BBWI', 'BDX', 'ZBH', 'SRE', 'MMM', 'IBM', 'T', 'CTAS', 'DECK', 'SMCI', 'LRCX', 'J', 'TSCO',
                          'ETR', 'LEN', 'WDC', 'FAST', 'ORLY'])

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
        share classes as part of the index, the keys are additional shares ticker symbols and values
        are the first share class (typically class A). Three corporations in the index have class B or class C shares.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                             storage_options=headers)
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
    def get_sp500_historical_components(start=None):
        """
        Returns a dictionary whose keys are ticker symbols representing companies that were part of the S&P 500 Index
        at any time since 'start' and whose values are pairs representing the dates of inclusion and exclusion from
        the index. A start date of 'None' means the ticker was part of the index before 'start'. An end date of 'None'
        implies the ticker is still part of the index.

        :param start: an int, float, str, datetime, or date object designating the starting time for calculating the
                      history of the constituent components of the S&P 500 Index.
        """
        return Metrics.get_historical_components(USStockMarketMetrics.get_sp500_components()[0],
                                                 './stock_market/sp500_changes_since_2019.csv', start)

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
                                    1308070268, 1308414093, 1310805008, 1311384883, 1325192508],
                                   index=pd.DatetimeIndex(['2020-02-13', '2020-07-30', '2020-08-23', '2020-10-26',
                                                           '2021-02-16', '2021-04-22', '2021-07-26', '2021-10-27',
                                                           '2022-02-14', '2022-04-20', '2022-07-26', '2022-10-26',
                                                           '2023-02-13', '2023-04-25', '2023-07-26', '2023-10-24',
                                                           '2024-02-12', '2024-04-19', '2024-07-23']).map(last_bd)),
                'CERN': pd.Series([311937692, 304348600, 305381551, 306589898, 301317068, 294222760, 294098094],
                                  index=pd.DatetimeIndex(['2020-01-28', '2020-04-23', '2020-07-22', '2020-10-21',
                                                          '2021-04-30', '2021-10-25', '2022-04-26']).map(last_bd)),
                'CTLT': pd.Series([164697598, 170226514, 170341553, 170787238, 171188042, 179104173, 179213237,
                                   179895677, 179963589, 180090483, 180271741,
                                   180641272, 180737675, 180979849, 181511586],
                                  index=pd.DatetimeIndex(['2020-10-29', '2021-01-25', '2021-04-27', '2021-08-23',
                                                          '2021-10-26', '2022-01-25', '2022-04-26', '2022-08-25',
                                                          '2022-10-27', '2023-01-31', '2023-05-31', '2023-11-30',
                                                          '2024-01-31', '2024-04-25', '2024-10-28']).map(last_bd)),
                'CTXS': pd.Series([123450644, 123123572, 124167045, 124230000, 124722872, 126579926, 126885081],
                                  index=pd.DatetimeIndex(['2020-04-28', '2020-10-23', '2021-04-29', '2021-06-30',
                                                          '2021-11-01', '2022-04-27', '2022-07-18']).map(last_bd)),
                'CXO': pd.Series([201028695, 196706121, 196701580, 196707339, 196304640],
                                 index=pd.DatetimeIndex(['2019-10-28', '2020-02-14', '2020-04-27',
                                                         '2020-07-26', '2020-10-23']).map(last_bd)),
                'DFS': pd.Series([313468253, 308337638, 306300776, 306421150, 306496332, 306691722, 304887527,
                                  299468440, 293075754, 284903687, 280965096, 273171312, 273225765, 261934442,
                                  253946033, 249947996, 250057955, 250555294, 250599037, 251071540, 251226920,
                                  251605294, 251653378],
                                 index=pd.DatetimeIndex(['2019-10-25', '2020-02-21', '2020-02-24', '2020-07-22',
                                                         '2020-10-21', '2021-02-12', '2021-04-28', '2021-07-23',
                                                         '2021-10-22', '2022-02-18', '2022-04-22', '2022-07-22',
                                                         '2022-10-21', '2023-02-17', '2023-04-20', '2023-07-21',
                                                         '2023-10-20', '2024-02-16', '2024-04-26', '2024-07-26',
                                                         '2024-12-13', '2025-02-28', '2025-04-25']).map(last_bd)),
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
                                   317737778, 315639479, 313376417, 311278353, 6163000000, 6086000000, 5968000000,
                                   5801000000],
                                  index=pd.DatetimeIndex(['2020-01-27', '2020-04-21', '2020-07-23', '2020-10-22',
                                                          '2021-01-26', '2021-04-20', '2021-07-20', '2021-10-19',
                                                          '2022-01-25', '2022-04-19', '2022-07-15', '2022-07-22',
                                                          '2022-10-18', '2023-01-26', '2023-07-18']).map(last_bd)),
                'GOOGL': pd.Series([299895185, 300050444, 300471156, 300643829, 300737081, 300746844, 301084627,
                                    300809676, 300754904, 300763622, 300446626, 5996000000, 5973000000, 5956000000,
                                    5933000000],
                                   index=pd.DatetimeIndex(['2020-01-27', '2020-04-21', '2020-07-23', '2020-10-22',
                                                           '2021-01-26', '2021-04-20', '2021-07-20', '2021-10-19',
                                                           '2022-01-25', '2022-04-19', '2022-07-15', '2022-07-22',
                                                           '2022-10-18', '2023-01-26', '2023-07-18']).map(last_bd)),
                'INFO': pd.Series([392948672, 398916408, 396809671, 398358566, 398612292, 398841378, 399080370],
                                  index=pd.DatetimeIndex(['2019-12-31', '2020-02-29', '2020-05-31', '2020-08-31',
                                                          '2021-05-31', '2021-08-31', '2021-12-31']).map(last_bd)),
                'JWN': pd.Series([156346167, 157032858],
                                 index=pd.DatetimeIndex(['2020-03-11', '2020-05-30']).map(last_bd)),
                'KSU': pd.Series([90964664, 90980440],
                                 index=pd.DatetimeIndex(['2021-07-09', '2021-10-12']).map(last_bd)),
                'MRO': pd.Series([795849999, 790312454, 789439221, 789391520, 789075988, 788153032, 788398843,
                                  778536897, 730765163, 707690859, 677583502, 635067840, 629654204, 617604314,
                                  605687177, 585247358, 577197369, 564035502, 559383423, 559410316],
                                  index=pd.DatetimeIndex(['2020-02-14', '2020-04-30', '2020-07-31', '2020-10-31',
                                                          '2021-02-12', '2021-04-30', '2021-07-31', '2021-10-29',
                                                          '2022-02-11', '2022-04-29', '2022-07-29', '2022-10-31',
                                                          '2023-02-10', '2023-04-28', '2023-07-31', '2023-10-31',
                                                          '2024-02-16', '2024-04-30', '2024-07-31', '2024-10-31'])
                                 .map(last_bd)),
                'MXIM': pd.Series([266625382, 266695209, 267301195, 268363654, 268566248],
                                  index=pd.DatetimeIndex(['2020-04-17', '2020-08-10', '2020-10-15',
                                                          '2021-04-16', '2021-08-10']).map(last_bd)),
                'NBL': pd.Series([479698676, 479768764],
                                 index=pd.DatetimeIndex(['2020-03-31', '2020-06-30']).map(last_bd)),
                'NLSN': pd.Series([356475591, 358497131, 359941875],
                                  index=pd.DatetimeIndex(['2020-03-31', '2021-03-31', '2022-09-30']).map(last_bd)),
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
        Based on rebalancing in H1 2025.
        """
        return ['INGA.AS', 'ABN.AS', 'DBK.DE', 'CBK.DE', 'BNP.PA', 'ACA.PA', 'GLE.PA', 'KBC.BR', 'BIRG.IR', 'A5G.IR',
                'BBVA.MC', 'BKT.MC', 'CABK.MC', 'SAB.MC', 'SAN.MC', 'EBS.VI', 'BG.VI', 'RBI.VI', 'NDA-FI.HE',
                'ISP.MI', 'UCG.MI', 'FBK.MI', 'BAMI.MI', 'BPE.MI', 'BMPS.MI', 'BPSO.MI', 'BGN.MI',
                'HSBA.L', 'BARC.L', 'LLOY.L', 'NWG.L', 'INVP.L', 'STAN.L',
                'BCVN.SW', 'CMBN.SW',
                'SEB-A.ST', 'SWED-A.ST', 'SHB-A.ST', 'AZA.ST',
                'DANSKE.CO', 'SYDB.CO', 'JYSK.CO', 'RILBA.CO',
                'DNB.OL', 'SB1NO.OL',
                'PEO.WA', 'PKO.WA', 'SPL.WA']


class NLStockMarketMetrics(EuropeBanksStockMarketMetrics):
    def __init__(self, tickers, additional_share_classes=None, stock_index='^AEX', start=None, hist_shares_outs=None,
                 currency_conversion_df=None):
        super().__init__(tickers, additional_share_classes, stock_index, start, hist_shares_outs,
                         currency_conversion_df)

    @staticmethod
    def get_aex_components():
        # URL of the AEX index Wikipedia page
        url = 'https://en.wikipedia.org/wiki/AEX_index'
        try:
            # Fetch all tables from the Wikipedia page
            tables = pd.read_html(url)
            #print(f"Found {len(tables)} tables on the page")
        except Exception as e:
            #print(f"Error fetching tables: {e}")
            return None

            # Keywords to identify the correct table
        keywords = ['company', 'sector', 'ticker', 'weight']

        # Iterate through each table
        for i, df in enumerate(tables):
            # Check if all column names are strings
            if not all(isinstance(col, str) for col in df.columns):
                #print(f"Skipping table {i} due to non-string columns: {df.columns.tolist()}")
                continue

            # Look for columns matching our keywords (case-insensitive)
            # print(f"Table {i} columns: {df.columns.tolist()}")
            matching_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in keywords)]

            # If at least 3 columns match, assume this is the composition table
            if len(matching_cols) >= 3:
                return  [component + '.AS' for component in df.loc[:, 'Ticker symbol']]

        return None

    @staticmethod
    def get_aex_historical_components(start=None):
        return Metrics.get_historical_components(NLStockMarketMetrics.get_aex_components(),
            './stock_market/aex_changes_since_2021.csv', start)
