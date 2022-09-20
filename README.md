# US Economic Data Analysis
This repository contains Jupyter notebooks that visually analyze US Economic data as provided by [St. Louis Fed](https://fred.stlouisfed.org).
The analysis is carried out using [Pandas](https://pandas.pydata.org), [Pandas datareader](https://pydata.github.io/pandas-datareader/), and [Matplotlib](https://matplotlib.org/stable/index.html).

So far I created the following notebooks:
* [Analysis of the evolution of the seasonally adjusted CPI, Fed Funds Effective Rate, Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, and comparison of the present US Treasury Yield Curve with Annual Inflation Expectations](./CPI_and_Fed_Funds_Rates.ipynb), plus [a similar analysis for the Eurozone by way of comparison](./CPI_and_ECB_Rates.ipynb)
* [Analysis of the evolution of the US Federal Public Debt as percentage of GDP, Public Debt as percentage of Federal annual tax revenue, Annual pulic deficit as percentage of GDP, and interest outlays to tax revenues](./Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb)
* [Analysis of the evolution of the ownership structure of US Federal Debt](./Fed_Public_Debt_Holders.ipynb)
* [Analysis of changes in M2, Real Personal Consumption Expenditures (PCE), Wage Inflation and CPI](./M2_PCE_and_CPI.ipynb)
* [Analysis of Participation, Employment to Population, Unemployment, and Unfilled Vacancies to Population Rates](./Unemployment_and_Participation_Rates.ipynb)

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything and run these notebooks in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```

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

A full run of these notebooks can be seen [here for CPI, Fed Funds Rate, Treasury rates and Inflation expectations](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/CPI_and_Fed_Funds_Rates.ipynb),
[here for public debt analysis](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb),
[here for public debt ownership analysis](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_Holders.ipynb),
[here for the analysis of M2, Real PCE, Wage Infation, and CPI](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/M2_PCE_and_CPI.ipynb), and
[here for the analysis of Participation, Employment to Population, Unemployment, and Unfilled Vacancies to Population Rates](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Unemployment_and_Participation_Rates.ipynb).

You can also run these notebooks in Google cloud. This way you don't need to install anything locally. This takes just a few seconds:
1. Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in your browser
2. In the modal window that appears select `GitHub`
3. Enter the URL of this repository's notebook, e.g.: `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb`
or
`https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/CPI_and_Fed_Funds_Rates.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_Holders.ipynb`
5. Click the search icon
6. Enjoy
