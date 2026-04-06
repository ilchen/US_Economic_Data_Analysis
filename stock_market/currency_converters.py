import pandas as pd

import yfinance as yfin

from stock_market import metrics


class EuroCurrencyConverter:
    """
    Converter of arbitrary currency amounts into Euros.
    """

    EXCHANGE_CUR_MAP = {
        '.L': 'GBp', '.SW': 'CHF', '.ST': 'SEK', '.CO': 'DKK', '.OL': 'NOK', '.PR': 'CZK', '.WA': 'PLN'
    }
    CURRENCY_MAP = {
        'USD': 'EUR=X', 'GBp': 'GBPEUR=X', 'CHF': 'CHFEUR=X', 'SEK': 'SEKEUR=X', 'DKK': 'DKKEUR=X', 'NOK': 'NOKEUR=X',
        'CZK': 'CZKEUR=X', 'PLN': 'PLNEUR=X'
    }
    INVERSE_MAP = {v: k for k, v in CURRENCY_MAP.items()}

    def __init__(self, currencies_for_conversion, start=None):
        """
        Constructs a dataframe to convert amounts from different currencies into the Euro using
        end of day exchange rates

        :param currencies_for_conversion:  a list of currencies whose prices will be converted
                                           into Euros using the following exchange rates:
                                        * USD -> EUR=X     (units of Euros for one US dollar)
                                        * GBp -> GBPEUR=X  (units of Euros for one British pound)
                                        * CHF -> CHFEUR=X  (units of Euros for one Swiss Franc)
                                        * SEK -> SEKEUR=X  (units of Euros for one Swedish Krona)
                                        * DKK -> DKKEUR=X  (units of Euros for one Danish Krone)
                                        * NOK -> NOKEUR=X  (units of Euros for one Norwegian Krone)
                                        * CZK -> CZKEUR=X  (units of Euros for one Czech Koruna)
                                        * PLN -> PLNEUR=X  (units of Euros for one Polish Zloty)
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses different kinds of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        """
        tickers = [EuroCurrencyConverter.CURRENCY_MAP[sfx] for sfx in currencies_for_conversion]
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
        if 'GBp' in currencies_for_conversion:
            self.cur_conv_df.loc[:, 'GBp'] /= 100.

    def get_currency_conversion_df(self):
        return self.cur_conv_df

    def exch_to_cur(suffixes_for_conversion):
        """
        Converts common exchange designator suffixes into their corresponding currency codes.
        
        @param suffixes_for_conversion: suffixes of ticker symbols (e.g. '.L', '.SW') representing exchanges they
                                        trade on
        @return: list of corresponding currencies (e.g. 'GBp', 'CHF')
        """
        return list(map(EuroCurrencyConverter.EXCHANGE_CUR_MAP.get, suffixes_for_conversion))
