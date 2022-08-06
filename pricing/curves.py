import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from pandas.tseries.offsets import BDay
from datetime import timedelta, date, time, datetime
from enum import Enum, unique
from math import sqrt, log, exp
from functools import reduce


@unique
class MaturityRepresentation(Enum):
    """
    A preferred index type for curve points returned by the :class:`~pricing.curves.YieldCurve` class.
    """

    PANDAS_TIMEDELTA = 0
    """
    Representation of maturities as pandas._libs.tslibs.timedeltas.Timedelta objects
    """

    DAYS = 1
    """
    Representation of maturities as numpy.int64 objects representing days
    """

    YEARS = 2
    """
    Representation of maturities as numpy.float64 objects representing years
    """


class YieldCurve:
    """
    A Yield curve defined by a list of {maturity, interest rate} pairs. This class uses Cubic splines by default
    to interpolate when constructing the curve. See <a href="http://web.math.ku.dk/~rolf/HaganWest.pdf">this article</a>
    for more details on interpolation methods.
    """

    def __init__(self, date, maturities, rates, k=3, align_on_business_days=True, compounding_freq=2):
        """
        Constructs a new curve based on specified maturities and rates.

        :param date: a datetime.date object relative to which the maturities are to be applied
        :param maturities: a list of relativedelta instances in increasing order
        :param rates: a list of corresponding yields in percent per annum
        :param k: degree of the smoothing spline for interpolation
        :param compounding_freq: how many times a year is the interest compounded, 0 implies continuous compounding
        """
        assert len(maturities) == len(rates) >= 2
        assert compounding_freq >= 0 and isinstance(compounding_freq, int)

        dt = datetime.combine(date, time())
        self.timestamps = [(dt + maturity + (BDay(0) if align_on_business_days else timedelta())).timestamp()
                           for maturity in maturities]

        # Verify it is monotonically increasing
        assert all(self.timestamps[i] <= self.timestamps[i + 1] for i in range(len(maturities) - 1))
        self.ppoly = InterpolatedUnivariateSpline(self.timestamps, rates, k=k)
        self.date = (date + BDay(0)).date() if align_on_business_days else date
        self.align_on_bd = align_on_business_days
        self.comp_freq = compounding_freq

    def get_curve_dates(self):
        """
        Returns curve dates for which yields were provided as datetime.date objects,
        possibly aligned with business dates.
        """
        return [date.fromtimestamp(timestamp) for timestamp in self.timestamps]

    def get_yield_for_maturity_date(self, date):
        """
        Returns the annual yield for maturity corresponding to 'date', possibly aligned on the next business day if 'date'
        is not a business date.

        :param date: a datetime.date object for which the yield needs to be calculated
        """
        timestamp = (datetime.combine(date, time()) + (BDay(0) if self.align_on_bd else timedelta())).timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        return self.ppoly(timestamp).tolist()

    def get_discount_factor_for_maturity_date(self, date):
        """
        Returns the discount factor for maturity corresponding to 'date', possibly aligned on the next business day
        if 'date' is not a business date.

        :param date: a datetime.date object for which the yield needs to be calculated
        :returns: a discount factor such that any cashflow on date 'date' should be multiplied
                  by value returned to obtain its NPV
        """
        adjusted_datetime = datetime.combine(date, time()) + (BDay(0) if self.align_on_bd else timedelta())
        timestamp = adjusted_datetime.timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        ytm = self.ppoly(timestamp).tolist()
        num_days = (adjusted_datetime.date() - self.date).days

        num_leap_years = YieldCurve.get_num_leap_years(self.date.year, adjusted_datetime.date().year)
        leap_years_add_on = num_leap_years / (adjusted_datetime.date().year - self.date.year)\
            if num_leap_years else 0
        num_years = num_days / (365. + leap_years_add_on)
        ytm = self.to_continuous_compounding(ytm)
        return exp(-ytm * num_years)

    def to_continuous_compounding(self, rate):
        return rate if self.comp_freq == 0 else self.comp_freq * log(1 + rate / self.comp_freq)

    def to_years(self, date):
        """
        Converts date to a maturity expressed in years relative to the starting date of this curve

        :param date: a datetime.date object that needs to be converted into maturity in years
        """

        adjusted_datetime = datetime.combine(date, time()) + (BDay(0) if self.align_on_bd else timedelta())
        timestamp = adjusted_datetime.timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        num_days = (adjusted_datetime.date() - self.date).days
        num_leap_years = YieldCurve.get_num_leap_years(self.date.year, adjusted_datetime.date().year)
        leap_years_add_on = num_leap_years / (adjusted_datetime.date().year - self.date.year) \
            if num_leap_years else 0
        num_years = num_days / (365. + leap_years_add_on)
        return num_years

    def get_yield_for_maturity_timestamp(self, timestamp):
        """
        Returns the annual yield for maturity corresponding to 'timestamp'

        :param timestamp: a POSIX timestamp (number of seconds since 1st Jan 1970 UTC).
        """
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        return self.ppoly(timestamp).tolist()

    def get_curve_points(self, n):
        """
        Returns a Series object corresponding to this yield curve's points evenly spaced,
        indexed by datatime.date values

        :param n: the number of points to return, must be >= 2
        """
        assert n >= 2
        delta = (self.timestamps[-1] - self.timestamps[0]) / (n - 1)
        timestamps = [self.timestamps[0] + i * delta for i in range(n)]
        pairs = list(zip(*[(date.fromtimestamp(timestamp), self.ppoly(timestamp).tolist())
                           for timestamp in timestamps]))
        return pd.Series(pairs[1], index=pairs[0], name=str(self.date))

    def get_curve_points_indexed_by_maturities(self, n, maturity_repr=MaturityRepresentation.PANDAS_TIMEDELTA):
        """
        Returns a Series object corresponding to this yield curve's points evenly spaced, indexed by maturities.

        :param n: the number of points to return, must be >= 2
        :param maturity_repr: an instance of the MaturityRepresentation enum designating the
                              preferred way to express maturities in a returned panda.Series
        """
        assert n >= 2
        delta = (self.timestamps[-1] - self.timestamps[0]) / (n - 1)
        timestamps = [self.timestamps[0] + i * delta for i in range(n)]
        pairs = list(zip(*[(date.fromtimestamp(timestamp) - self.date,
                            self.ppoly(timestamp).tolist()) for timestamp in timestamps]))
        ret = pd.Series(pairs[1], index=pairs[0], name=str(self.date))
        if maturity_repr != MaturityRepresentation.PANDAS_TIMEDELTA:
            ret = ret.set_axis(ret.index.days)
            if maturity_repr == MaturityRepresentation.YEARS:
                # Compensation for leap years, dividing the number of days by 365 will incorrectly represent
                # long maturities expressed in years
                last_year = date.fromtimestamp(timestamps[-1]).year
                num_leap_years = YieldCurve.get_num_leap_years(self.date.year, last_year)
                leap_years_add_on = num_leap_years / (last_year - self.date.year)
                num_years_index = ret.index / (365. + leap_years_add_on)
                ret = ret.set_axis(num_years_index)
        return ret

    @staticmethod
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    def get_num_leap_years(year_start, year_end):
        return reduce(lambda accu, year: accu + YieldCurve.is_leap_year(year),
                      range(year_start, year_end), 0)