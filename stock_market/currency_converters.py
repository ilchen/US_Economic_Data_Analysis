import pandas as pd

import yfinance as yfin

from stock_market import metrics


class EuroCurrencyConverter:
    """
    Converter of arbitrary currency amounts into Euros.
    """

    CURRENCY_MAP = {
        '.L': 'GBPEUR=X', '.SW': 'CHFEUR=X', '.ST': 'SEKEUR=X',
        '.CO': 'DKKEUR=X', '.OL': 'NOKEUR=X', '.PR': 'CZKEUR=X',
        '.WA': 'PLNEUR=X'
    }
    INVERSE_MAP = {v: k for k, v in CURRENCY_MAP.items()}

    def __init__(self, suffixes_for_conversion, start=None):
        """
        Constructs a dataframe to convert amounts from different currencies into the Euro using
        end of day exchange rates

        :param suffixes_for_conversion: suffixes of ticker symbols (e.g. '.L', '.SW') whose prices will be converted
                                        into Euros using the following exchange rates:
                                        * .L  -> GBPEUR=X  (units of Euros for one British pound)
                                        * .SW -> CHFEUR=X  (units of Euros for one Swiss Franc)
                                        * .ST -> SEKEUR=X  (units of Euros for one Swedish Krona)
                                        * .CO -> DKKEUR=X  (units of Euros for one Danish Krone)
                                        * .OL -> NOKEUR=X  (units of Euros for one Norwegian Krone)
                                        * .PR -> CZKEUR=X  (units of Euros for one Czech Koruna)
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses different kinds of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        """
        tickers = [EuroCurrencyConverter.CURRENCY_MAP[sfx] for sfx in suffixes_for_conversion]
        yf_tickers = yfin.Tickers(tickers)
        self.cur_conv_df = yf_tickers.download(start=start, auto_adjust=False, actions=False, ignore_tz=True)
        self.cur_conv_df = self.cur_conv_df.loc[:, metrics.Metrics.CLOSE]
        self.cur_conv_df.columns = [EuroCurrencyConverter.INVERSE_MAP[ticker] for ticker in self.cur_conv_df.columns]

        # Required until the 'ignore_tz' parameter in the 'download' method starts working again
        self.cur_conv_df = self.cur_conv_df.tz_localize(None)

        # Ensure coverage for all days
        missing_days = pd.DataFrame(
            index=pd.date_range(start, self.cur_conv_df.index[-1], freq='D').difference(self.cur_conv_df.index),
            columns=self.cur_conv_df.columns, dtype='float64')
        self.cur_conv_df = pd.concat([self.cur_conv_df, missing_days]).sort_index().ffill()

        # London stock exchange quotes in pence
        if '.L' in suffixes_for_conversion:
            self.cur_conv_df.loc[:, '.L'] /= 100.

    def get_currency_conversion_df(self):
        return self.cur_conv_df
