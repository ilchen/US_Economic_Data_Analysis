from enum import Enum

import requests

import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd


class MarketableSecurityType(Enum):
    TREASURY_NOTES = "Treasury Notes"
    TREASURY_BONDS = "Treasury Bonds"
    TREASURY_INFLATION_INDEXED_NOTES = "Treasury Inflation-Indexed Notes"
    TREASURY_INFLATION_INDEXED_BONDS = "Treasury Inflation-Indexed Bonds"
    FEDERAL_FINANCING_BANK = "Federal Financing Bank"
    TOTAL_MARKETABLE = "Total Marketable"
    TREASURY_BILLS = "Treasury Bills"
    TREASURY_TIPS = "Treasury Inflation-Protected Securities (TIPS)"
    TREASURY_FRN = "Treasury Floating Rate Notes (FRN)"


class MTS:
    """
    Monthly Treasury Statement data from the U.S. Department of the Treasury.
    """
    
    BASE_URL = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
    TABLE_9_ENDPOINT = '/v1/accounting/mts/mts_table_9'
    AVG_INTEREST_ENDPOINT = '/v2/accounting/od/avg_interest_rates'
    CLASSIFICATION_DESC_FIELD = 'classification_desc'
    CLASSIFICATION_DESC_NET_INTEREST = 'Net Interest'
    SECURITY_TYPE_DESC_FIELD = 'security_type_desc'
    SECURITY_TYPE_DESC_MARKETABLE = 'Marketable'
    TOTAL_MARKETABLE_ALT = "TotalMarketable"

    
    def __init__(self, start, end=None):
        """
        Constructs an MTS object bound to the specified reporting data range.
    
        :param start_date: a date of the first month for which data needs to be obtained
        :param end_date: a date of the last month for which data needs to be obtained
        """
        self.start_date = MonthBegin().rollback(start).date()
        self.end_date = (MonthEnd().rollforward(end) if end is not None else MonthEnd().rollforward(date.today())).date()

    def retrieve_net_interest(self):
        """
        Retrieves net interest paid by U.S. Department of the Treasury on its public debt. I used Table 9 of
        the Monthly Treasury Statement (MTS).
    
        :returns: a pd.Series object capturing net interest outlays of the U.S. Treasury in each month in US dollars
                  starting from 'self.start_date' and ending on 'self.end_date' including.
        """
        
        # Define parameters for the API request
        params = {
            'filter': f'record_date:gte:{self.start_date},record_date:lte:{self.end_date},'
                      f'{MTS.CLASSIFICATION_DESC_FIELD}:eq:{MTS.CLASSIFICATION_DESC_NET_INTEREST}',
            'page[size]': 1000,
            'fields': f'record_date,{MTS.CLASSIFICATION_DESC_FIELD},current_month_rcpt_outly_amt'
        }
    
        # Make the API request
        response = requests.get(MTS.BASE_URL + MTS.TABLE_9_ENDPOINT, params=params)
    
        # Check if the request was successful
        response.raise_for_status()  # Raises HTTPError for bad requests (4XX, 5XX)
    
        # Parse the JSON response
        data = response.json()
    
        # Check if there is any data in the response
        if data['data']:
            # Convert the JSON data to a Pandas DataFrame
            df = pd.DataFrame(data['data'])
            df = df.set_index('record_date')
            df = df.set_axis(pd.DatetimeIndex(df.index, 'ME'))

            # Convert into an appropriately named pd.Series object
            return df.loc[:, 'current_month_rcpt_outly_amt'].rename(MTS.CLASSIFICATION_DESC_NET_INTEREST).astype('float64')
        else:
            raise KeyError(f'Date range [{start_date}; {end_date}] not present in the dataset.')

    def retrieve_avg_interest(self, security_type):
        """
        Retrieves the average interest that U.S. Department of the Treasury paid on marketable securities in its public debt.
    
        :param security_type: a MarketableSecurityType enum member or a list of MarketableSecurityType enum members
        :returns: a pd.Series or pd.DataFrame object capturing the average interest that the U.S. Treasury paid in each month 
                  starting from 'self.start_date' and ending on 'self.end_date' including for security types captured by
                  the 'security_type' parameter
        """

        if isinstance(security_type, list):
            if len(security_type) == 0:
                return pd.Series()
            security_type = list(set(security_type))
            assert len(security_type) <= len(MarketableSecurityType)
            assert isinstance(security_type[0], MarketableSecurityType)
        else:
            assert isinstance(security_type, MarketableSecurityType)
            security_type = [security_type]

        # Define parameters for the API request
        params = {
            'filter': f'record_date:gte:{self.start_date},record_date:lte:{self.end_date},'
                      f'{MTS.SECURITY_TYPE_DESC_FIELD}:eq:{MTS.SECURITY_TYPE_DESC_MARKETABLE}',
            'page[size]': 10000,
            'fields': f'record_date,{MTS.SECURITY_TYPE_DESC_FIELD},security_desc,avg_interest_rate_amt'
        }
        
        # Make the API request
        response = requests.get(MTS.BASE_URL + MTS.AVG_INTEREST_ENDPOINT, params=params)
    
        # Check if the request was successful
        response.raise_for_status()  # Raises HTTPError for bad requests (4XX, 5XX)
    
        # Parse the JSON response
        data = response.json()
    
        # Check if there is any data in the response
        if data['data']:
            # Convert the JSON data to a Pandas DataFrame
            df = pd.DataFrame(data['data'])
            df = df.set_index('record_date')
            df = df.pivot(columns='security_desc', values='avg_interest_rate_amt')
            #df = df.set_axis(pd.DatetimeIndex(df.index))

            if MTS.TOTAL_MARKETABLE_ALT in df.columns and MarketableSecurityType.TOTAL_MARKETABLE in security_type:
                idx = ~df.loc[:, MTS.TOTAL_MARKETABLE_ALT].astype('float64').isna()
                df.loc[idx, MarketableSecurityType.TOTAL_MARKETABLE.value] = df.loc[idx, MTS.TOTAL_MARKETABLE_ALT]

            ret = df.loc[:, [sec.value for sec in security_type]].astype('float64').set_axis(pd.DatetimeIndex(df.index, 'ME'))
            return ret if len(security_type) > 1 else ret.squeeze()
            
        else:
            raise KeyError(f'Date range [{start_date}; {end_date}] not present in the dataset.')
