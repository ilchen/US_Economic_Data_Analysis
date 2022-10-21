import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from pandas.tseries.offsets import BDay
from datetime import timedelta, date, time, datetime
from enum import Enum, unique
from math import log, exp
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
        :param maturities: a list of relativedelta instances in increasing order or
                           a list of timestamps represented as floats
        :param rates: a list or a numpy.ndarray of corresponding yields in percent per annum, points with
                      numpy.nan values are discarded
        :param k: degree of the smoothing spline for interpolation
        :param align_on_business_days: designates if 'date' and other date values need to be aligned to a business
                                       date
        :param compounding_freq: how many times a year is the interest compounded, 0 implies continuous compounding
        """
        assert len(maturities) == len(rates) >= 2
        assert compounding_freq >= 0 and isinstance(compounding_freq, int)

        dt = datetime.combine(date, time())
        if isinstance(maturities[0], float):
            self.timestamps = list(maturities)
        else:
            self.timestamps = [(dt + maturity + (BDay(0) if align_on_business_days else timedelta())).timestamp()
                               for maturity in maturities]

        # Verify it is monotonically increasing
        assert all(self.timestamps[i] <= self.timestamps[i + 1] for i in range(len(maturities) - 1))

        # Discard numpy.nan datapoints
        mask = np.logical_not(np.isnan(rates))
        self.ppoly = InterpolatedUnivariateSpline(np.array(self.timestamps)[mask], np.array(rates)[mask], k=k)
        self.date = (date + BDay(0)).date() if align_on_business_days else date
        self.align_on_bd = align_on_business_days
        self.comp_freq = compounding_freq

    def get_curve_dates(self):
        """
        Returns curve dates for which yields were provided as datetime.date objects,
        possibly aligned with business dates.
        """
        return [date.fromtimestamp(timestamp) for timestamp in self.timestamps]

    def get_yield_for_maturity_date(self, dt):
        """
        Returns the annual yield for maturity corresponding to 'date', possibly aligned on the next business day if 'dt'
        is not a business date.

        :param dt: a datetime.date object for which the yield needs to be calculated
        """
        timestamp = (datetime.combine(dt, time()) + (BDay(0) if self.align_on_bd else timedelta())).timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        return self.ppoly(timestamp).tolist()

    def get_discount_factor_for_maturity_date(self, dt):
        """
        Returns the discount factor for maturity corresponding to 'dt', possibly aligned on the next business day
        if 'dt' is not a business date.

        :param dt: a datetime.date object for which the yield needs to be calculated
        :returns: a discount factor such that any cashflow on date 'dt' should be multiplied
                  by value returned to obtain its NPV
        """
        adjusted_datetime = datetime.combine(dt, time()) + (BDay(0) if self.align_on_bd else timedelta())
        timestamp = adjusted_datetime.timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]
        ytm = self.ppoly(timestamp).tolist()
        num_years = YieldCurve.year_difference(self.date, adjusted_datetime.date())
        ytm = self.to_continuous_compounding(ytm)
        return exp(-ytm * num_years)

    def get_forward_discount_factor_for_maturity_date(self, forward_datetime, dt):
        """
        Returns the discount factor for maturity corresponding to 'dt' relative to the forward date 'forward_datetime'
        :param forward_datetime: a datetime.datetime object relative to which the discount factor
                                 for maturity 'dt' needs to be calculated
        :param dt: a datetime.date or datetime.datetime object for which the yield needs to be calculated
        :returns: a discount factor such that any cashflow on date 'date' should be multiplied
                  by value returned to obtain its NPV
        """
        adjusted_datetime = datetime.combine(dt, time()) if type(dt) is date else dt
        timestamp = adjusted_datetime.timestamp()
        forward_timestamp = forward_datetime.timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]\
               and self.timestamps[0] <= forward_timestamp <= self.timestamps[-1]\
               and forward_timestamp < timestamp
        ytm = self.ppoly(timestamp).tolist()
        ytf = self.ppoly(forward_timestamp).tolist()
        num_years_to_maturity = YieldCurve.year_difference(self.date, adjusted_datetime)
        num_years_to_forward = YieldCurve.year_difference(self.date, forward_datetime)
        yfw = (ytm * num_years_to_maturity - ytf * num_years_to_forward)\
              / (num_years_to_maturity - num_years_to_forward)
        yfw = self.to_continuous_compounding(yfw)
        return exp(-yfw * (num_years_to_maturity - num_years_to_forward))

    def to_continuous_compounding(self, rate):
        return rate if self.comp_freq == 0 else self.comp_freq * log(1 + rate / self.comp_freq)

    def to_years(self, dt):
        """
        Converts 'dt' to a maturity expressed in years relative to the starting date of this curve
        :param dt: a datetime.date object that needs to be converted into maturity in years
        """

        adjusted_datetime = datetime.combine(dt, time()) + (BDay(0) if self.align_on_bd else timedelta())
        timestamp = adjusted_datetime.timestamp()
        assert self.timestamps[0] <= timestamp <= self.timestamps[-1]

        # An alternative less accurate but simpler way of calculating the same
        # return (pd.to_datetime(adjusted_datetime) - pd.to_datetime(self.date)) / np.timedelta64(1, 'Y')

        num_days = (adjusted_datetime.date() - self.date).days
        num_leap_years = YieldCurve.get_num_leap_years(self.date.year, adjusted_datetime.date().year)
        leap_years_add_on = num_leap_years / (adjusted_datetime.date().year - self.date.year) \
            if num_leap_years else 0
        num_years = num_days / (365. + leap_years_add_on)
        return num_years

    def to_timedelta(self, delta_in_years):
        """
        Converts delta_in_years relative to the starting date of this curve into a datetime.timedelta object

        :param delta_in_years: a float value designating time in years from the starting date of this curve
        """
        num_leap_years = YieldCurve.get_num_leap_years(self.date.year, int(self.date.year + delta_in_years))
        return timedelta(minutes=int((365 + num_leap_years) * 24 * 60 * delta_in_years))

    def to_datetime(self, delta_in_years):
        """
        Converts delta_in_years relative to the starting date of this curve into a datetime.datetime object

        :param delta_in_years: a float value designating time in years from the starting date of this curve
        """
        return datetime.combine(self.date, time()) + self.to_timedelta(delta_in_years) if delta_in_years != 0.\
            else datetime.fromtimestamp(self.timestamps[0])

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
                leap_years_add_on = 0 if num_leap_years == 0 else num_leap_years / (last_year - self.date.year)
                num_years_index = ret.index / (365. + leap_years_add_on)
                ret = ret.set_axis(num_years_index)
        return ret

    def parallel_shift(self, basis_points):
        """
        Returns a new YieldCurve object initialized analogously to this curve except that all yields are shifted
        by the specified amount of basis points.

        :param basis_points: the number of basis points to add to the yields of this curve
        :return: a new YieldCurve object
        """
        rates = np.array([self.ppoly(ts).tolist() for ts in self.timestamps]) + basis_points * 1e-4
        spline_degree = self.ppoly.__dict__['_data'][5]
        return YieldCurve(self.date, self.timestamps, rates, spline_degree,
                          align_on_business_days=self.align_on_bd, compounding_freq=self.comp_freq)

    @staticmethod
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    def get_num_leap_years(year_start, year_end):
        return reduce(lambda accu, year: accu + YieldCurve.is_leap_year(year),
                      range(year_start, year_end), 0)

    @staticmethod
    def year_difference(date1, date2):
        assert isinstance(date1, (date, datetime)) and isinstance(date2, (date, datetime))
        # Normalize to datetime
        date1 = date1 if isinstance(date1, datetime) else datetime.combine(date1, time())
        date2 = date2 if isinstance(date2, datetime) else datetime.combine(date2, time())
        time_delta = date2 - date1
        num_leap_years = YieldCurve.get_num_leap_years(date1.year, date2.year)
        leap_years_add_on = 0 if num_leap_years == 0 else num_leap_years / (date2.year - date1.year)
        return (time_delta.days + time_delta.seconds / (24. * 60 * 60)) / (365. + leap_years_add_on)
