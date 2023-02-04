# US Economic Data Analysis
This repository contains Jupyter notebooks that visually analyze US Economic data as provided by [St. Louis Fed](https://fred.stlouisfed.org).
The analysis is carried out using [Pandas](https://pandas.pydata.org), [Pandas datareader](https://pydata.github.io/pandas-datareader/), and [Matplotlib](https://matplotlib.org/stable/index.html).

So far I created the following notebooks:
* [Analysis of the evolution of the seasonally adjusted CPI, Fed Funds Effective Rate, Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, and comparison of the present US Treasury Yield Curve with Annual Inflation Expectations](./CPI_and_Fed_Funds_Rates.ipynb), plus [a similar analysis for the Eurozone by way of comparison](./CPI_and_ECB_Rates.ipynb)
* [Analysis of the evolution of the US Federal Public Debt as percentage of GDP, Public Debt as percentage of Federal annual tax revenue, Annual pulic deficit as percentage of GDP, and interest outlays to tax revenues](./Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb)
* [Analysis of the evolution of the ownership structure of US Federal Debt](./Fed_Public_Debt_Holders.ipynb)
* [Analysis of changes in M2, Real Personal Consumption Expenditures (PCE), Wage Inflation and CPI](./M2_PCE_and_CPI.ipynb)
* [Analysis of Participation, Employment to Population, Unemployment, and Unfilled Vacancies to Population Rates](./Unemployment_and_Participation_Rates.ipynb)
* [Analysis of US Money Supply](./Money_Supply.ipynb)
* [Analysis of US Interest Rate Spreads](./Interest_Rate_Spreads.ipynb) 
* [Analysis of US Past, Current, and Future Riskfree Rates](./Current_Riskfree_Rates.ipynb)

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything and run these notebooks in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```
**NB**: I use Yahoo-Finance data in the `Current_Riskfree_Rates.ipynb` notebook. Unfortunately Yahoo recently changed their API, as a result the last official version of pandas-datareader fails when retrieving data from Yahoo-Finance. To overcome it, until a new version of pandas-datareader addresses this, I added a dependency on yfinance and adjusted the notebook to make a `yfin.pdr_override()`.

## How to run
After you clone the repo and `cd` into its directory and run one of the below commands depending on which notebook you are interested in:
```commandline
jupyter notebook CPI_and_Fed_Funds_Rates.ipynb
```
or
```commandline
jupyter notebook Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb
```
or
```commandline
jupyter notebook Fed_Public_Debt_Holders.ipynb
```
or
```commandline
jupyter notebook M2_PCE_and_CPI.ipynb
```
or
```commandline
jupyter notebook Unemployment_and_Participation_Rates.ipynb
```
or
```commandline
jupyter notebook Money_Supply.ipynb
```
or
```commandline
jupyter notebook Interest_Rate_Spreads.ipynb
```
or
```commandline
jupyter notebook Current_Riskfree_Rates.ipynb
```

A full run of these notebooks can be seen [here for CPI, Fed Funds Rate, Treasury rates and Inflation expectations](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/CPI_and_Fed_Funds_Rates.ipynb),
[here for public debt analysis](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb),
[here for public debt ownership analysis](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_Holders.ipynb),
[here for the analysis of M2, Real PCE, Wage Infation, and CPI](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/M2_PCE_and_CPI.ipynb),
[here for the analysis of Participation, Employment to Population, Unemployment, and Unfilled Vacancies to Population Rates](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Unemployment_and_Participation_Rates.ipynb),
[here for the analysis of US Money supply](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Money_Supply.ipynb), and
[here for the analysis of US Interest Rate Spreads](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Interest_Rate_Spreads.ipynb), and
[here for the analysis of US Past, Current, and Future Riskfree rates](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Current_Riskfree_Rates.ipynb).

You can also run these notebooks in Google cloud. This way you don't need to install anything locally. This takes just a few seconds:
1. Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in your browser
2. In the modal window that appears select `GitHub`
3. Enter the URL of this repository's notebook, e.g.: `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/CPI_and_Fed_Funds_Rates.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_Holders.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/M2_PCE_and_CPI.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Unemployment_and_Participation_Rates.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Money_Supply.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Interest_Rate_Spreads.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Current_Riskfree_Rates.ipynb`
5. Click the search icon
6. Enjoy
