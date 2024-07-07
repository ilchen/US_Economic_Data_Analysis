from datetime import date, datetime

from math import exp

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, QuarterBegin
import yfinance as yfin

from scipy.optimize import minimize_scalar

from dateutil.relativedelta import relativedelta


class CMEFixedIncomeFuturesRates:
    """
    Base class for inferring rates and yields from interest rates futures
    """

    # CME's convention for months starting from January
    MONTHS = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    ADJ_CLOSE = 'Adj Close'

    def __init__(self, cur_date):
        """
        Constructs an instance to infer future rates from the prices of futures contracts.

        :param cur_date: a datetime.date or pandas.Timestamp object specifying the current month, relative to which
                         future rates are to be calculated
        """
        assert isinstance(cur_date, (date, datetime, pd.Timestamp))
        self.cur_date = cur_date.date() if isinstance(cur_date, (datetime, pd.Timestamp)) else cur_date

    def get_next_n_months_tickers(self, n, ticker_prefix='ZQ', ticker_suffix='.CBT'):
        """
        Returns CME ticker symbols for the next n months

        :param n: an integer indicating how many future months to return ticker symbols for
        :param ticker_prefix: a string designating a CME's ticker symbol prefix
        :param ticker_suffix: a string designating a suffix for the ticker symbols returne
        :returns: a list of tuples where the first tuple component is the ticket symbol represented as a string
                  and the second is a date.date object rounded down to the start of the month
        """
        months = [self.cur_date + relativedelta(months=+i) for i in range(1, n + 1)]
        return [(ticker_prefix + self.MONTHS[m.month - 1] + str(m.year)[-2:] + ticker_suffix, date(m.year, m.month, 1))
                for m in months]

    def get_next_n_quarter_tickers(self, n, ticker_prefix='ZN', ticker_suffix='.CBT'):
        """
        Returns CME ticker symbols for the next n quarters.

        :param n: and integer indicating how many future quarters to return ticker symbols
        :param ticker_prefix: a string designating a CME's ticker symbol prefix
        :param ticker_suffix: a string designating a suffix for the ticker symbols returne
        :returns: a list of tuples where the first tuple component is the ticket symbol represented as a string
                  and the second is a date.date object rounded down to the start of the month
        """
        months = [self.cur_date + i * QuarterBegin(startingMonth=3) for i in range(1, n + 1)]
        return [(ticker_prefix + self.MONTHS[m.month - 1] + str(m.year)[-2:] + ticker_suffix, date(m.year, m.month, 1))
                for m in months]

    @staticmethod
    def from_actual_360_to_actual_actual(series):
        """
        Converts the rates stored in a pandas.Series object from actual/360 day count convention to actual/actual.

        :param series: a pandas.Series object indexed by pandas.DatetimeIndex whose rates are to be converted
        :returns: a new pandas.Series object with rates using the actual/actual day count convention
        """
        # Converting the Fed Funds Rate to actual/actual
        leap_year_cond = series.index.year % 4 == 0 & ((series.index.year % 100 != 0) | (series.index.year % 400 == 0))
        ret = series.copy()
        ret[leap_year_cond] *= 366. / 360
        ret[np.invert(leap_year_cond)] *= 365. / 360
        return ret

    @staticmethod
    def from_continuous_compound_to_semiannual(series):
        """
        Converts the rates stored in a pandas.Series object from continuous compounding frequency to semiannual.

        :param series: a pandas.Series object indexed by pandas.DatetimeIndex whose rates are to be converted
        :returns: a new pandas.Series object with rates using semiannual compounding frequency
        """
        return 2. * (np.exp(series / 2.) - 1.)

    @staticmethod
    def tnote_price_to_yield(tnote_price, maturity=7):
        """
        Converts an n-year T-Note/Bond futures price to a corresponding yield with semiannual compounding.
        The 10-year T-Note contract allows for delivery of any T-Note with fixed semi-annual coupons and
        a remaining time to maturity of no less than 6.5 years and no more than 7.75 years.

        :param tnote_price: a float64 value representing the T-Note/Bond price
        :param maturity: an integer value representing the maturity of the T-Note/Bond that is expected to be delivered
        """

        # CME T-Note/Bond futures contracts are priced assuming a 6% par yield and 6% yield to maturity
        par_yield_tnote = CashflowDescriptor(.06, 2, 100, maturity)
        objective_func = lambda y: abs(tnote_price - par_yield_tnote.pv_all_cashflows(y))

        # Looking for yields in the range of 0%-20%
        res = minimize_scalar(objective_func, bounds=(0, .20), method='bounded')

        if res.success:
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nfev))
            return res.x
        else:
            raise ValueError("Optimizing the objective function with the passed T-Note price changes didn't succeed")


class CMEFedFundsFuturesRates(CMEFixedIncomeFuturesRates):
    """
    This class infers future Fed Funds effective rates from Fed Funds Futures traded on CME. Rates returned
    use the Actual/360 day count convention.
    """

    def __init__(self, cur_date):
        super().__init__(cur_date)

    def get_rates_for_next_n_months(self, n, dt=None):
        """
        Returns a pandas.Series object indexed by pandas.DatetimeIndex whose values represent
        the average future Fed Funds rate for a given future month

        :param n: for how many months (starting from the next month from the date this instance was initialized with)
                  to return the average future Fed Funds rates
        :param dt: datetime.date or pandas.Timestamp object specifying a past business day to use for retrieving
                   the prices of Fed Funds Futures contracts, if set to None the current date this instance was
                   initialized with will be used.
        """
        tickers, months = list(zip(*self.get_next_n_months_tickers(n)))
        dt = dt.date() if isinstance(dt, (datetime, pd.Timestamp)) else dt if isinstance(dt, date) else self.cur_date
        series = yfin.Tickers(list(tickers)).download(start=dt - BDay(3), end=dt,  auto_adjust=False, actions=False,
                                                      ignore_tz=True).loc[:, self.ADJ_CLOSE]
        if len(series) > 0:
            series = series.iloc[-1]
        return ((100. - series.reindex(tickers)) / 100.).set_axis(pd.DatetimeIndex(months))


class CME10YearTNoteFuturesYields(CMEFixedIncomeFuturesRates):
    """
    This class infers future 10-Year T-Note yields traded on CME. Rates returned use the Actual/Actual day count
    convention and a semi-annual compounding frequency. The yields returned are an approximation of correct future
    10-Year T-Note yields given that CME allows delivery of T-Notes with maturity of 6.5 years and more.
    """

    def __init__(self, cur_date):
        super().__init__(cur_date)

    def get_yields_for_next_n_quarters(self, n, dt=None):
        """
        Returns a pandas.Series object indexed by pandas.DatetimeIndex whose values represent
        the average future Fed Funds rate for a given future month
        """
        tickers, months = list(zip(*self.get_next_n_quarter_tickers(n)))
        dt = dt.date() if isinstance(dt, (datetime, pd.Timestamp)) else dt if isinstance(dt, date) else self.cur_date
        series = yfin.Tickers(list(tickers)).download(start=dt - BDay(3), end=dt,  auto_adjust=False, actions=False,
                                                      ignore_tz=True).loc[:, self.ADJ_CLOSE]
        if len(series) > 0:
            series = series.iloc[-1]
        series = series.reindex(tickers).set_axis(pd.DatetimeIndex(months)).dropna()
        series2 = series.apply(self.tnote_price_to_yield)
        return self.from_continuous_compound_to_semiannual(series2)


class CashflowDescriptor:
    """
    Represents cashflow schedules
    """

    def __init__(self, coupon_rate, coupon_frequency, notional, T):
        """
        :param coupon_rate: coupon rate per annum
        :param coupon_frequency: how many times a year is coupon paid
        :param notional: notional amount due
        :param T: time when the last coupon and notional are due
        """
        self.coupon_rate = coupon_rate
        self.coupon_frequency = coupon_frequency
        self.notional = notional
        self.T = T
        self.coupon_interval = 1 / coupon_frequency
        self.timeline = np.arange(self.coupon_interval, T + self.coupon_interval, self.coupon_interval)
        self.coupon_amount = notional * coupon_rate * self.coupon_interval

    def cashflow(self, t, coupon_rate=None):
        """
        :param t: future time in years
        :param coupon_rate: a different coupon rate from self.coupon_rate, might be handy when valuing asset swaps
        """
        coupon_amount = self.coupon_amount if coupon_rate is None else self.notional * coupon_rate * self.coupon_interval
        if t in self.timeline:
            return coupon_amount + (self.notional if t == self.T else 0)
        else:
            return 0

    def pv_cashflows_from_time(self, start_time, discount_rate):
        """
        Calculates the value of cashflows past 'start_time' as seen at 'start_time'
        """
        start = start_time if start_time in self.timeline else self.timeline[self.timeline.searchsorted(start_time)]
        timeline = np.arange(start, self.T + self.coupon_interval, self.coupon_interval)
        return self.pv_cashflows(timeline, discount_rate, t0=start_time)

    def pv_cashflows(self, timeline, discount_rate, t0=0):
        return sum(map(lambda t: self.cashflow(t) * exp(-discount_rate * (t - t0)), timeline))

    def pv_all_cashflows(self, discount_rate, t0=0):
        return self.pv_cashflows(self.timeline, discount_rate, t0)

    # Special method needed to value the floating leg of asset swaps
    def pv_all_cashflows_with_other_coupon_rate(self, other_coupon_rate, discount_rate, t0=0):
        return sum(map(lambda t: self.cashflow(t, other_coupon_rate) * exp(-discount_rate * (t - t0)), self.timeline))

