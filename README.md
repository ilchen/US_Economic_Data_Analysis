# US Economic Data Analysis
This repository contains Jupyter notebooks that visually analyze US Economic data as provided by [St. Louis Fed](https://fred.stlouisfed.org), the [OECD](https://stats.oecd.org), and [Yahoo-Finance](https://finance.yahoo.com/). The analysis is carried out using [Pandas](https://pandas.pydata.org), [Pandas datareader](https://pydata.github.io/pandas-datareader/), and [Matplotlib](https://matplotlib.org/stable/index.html).

So far I created the following notebooks (following a given link lets you see the most recent run of its notebook, I aim to refresh results monthly):
* [Analysis of CPI, Fed Funds Rate, Treasury rates and Inflation expectations](./CPI_and_Fed_Funds_Rates.ipynb), plus [a similar analysis for the Eurozone by way of comparison](./HICP_and_ECB_Rates.ipynb)
* [Analysis of the evolution of the US Federal Public Debt](./Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb)
* [Analysis of the evolution of the ownership structure of US Federal Debt](./Fed_Public_Debt_Holders.ipynb)
* [Analysis of changes in M2, Real Personal Consumption Expenditures (PCE), Wage Inflation and CPI](./M2_PCE_and_CPI.ipynb)
* [Analysis of Participation, Employment, Unemployment, Job-vacancy, and Unfilled Vacancies to Population Rates](./Unemployment_and_Participation_Rates.ipynb), plus [a similar analysis for the Eurozone by way of comparison](./Unemployment_and_Participation_Rates_Eurozone.ipynb)
* [Analysis of US Money Supply](./Money_Supply.ipynb), plus [a similar analysis for the Eurozone by way of comparison](./Money_Supply_Eurozone.ipynb)
* [Analysis of Quantitative Easing and Tapering by the Federal Reserve](./Quantitative_Easing_and_Tapering.ipynb)
* [Analysis of US Treasury Yields' Spreads](./Interest_Rate_Spreads.ipynb) 
* [Analysis of US Past, Current, and Future Riskfree Rates](./Current_Riskfree_Rates.ipynb), plus [a similar analysis for the Eurozone](./Current_Riskfree_Rates_Eurozone.ipynb)
* [Analysis of US Industrial Production](./Industrial_Production.ipynb)
* [Analysis of US GDP, its composition by industry, and trends in its make-up](./GDP_Composition.ipynb), plus [a similar analysis for the Eurozone](./GDP_Composition_Eurozone.ipynb)
* [Analysis of US Labor productivity (incl. comparison with that in EU)](./Labor_Productivity.ipynb)
* [Analysis of disposable income and savings rates of US individuals](./Disposable_Income_and_Savings.ipynb), plus [a similar analysis for the Eurozone](./Disposable_Income_and_Savings_Eurozone.ipynb)
* [Analysis of US Stock Market, including various metrics on the S&P 500 Index](./Stock_Market.ipynb), plus [a similar analysus for Stoxx Europe 600 Banks Index](./Stock_Market_Stoxx_Europe_Banks.ipynb)

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything and run these notebooks in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```
**NB**: I use Yahoo-Finance data in the `Current_Riskfree_Rates.ipynb` notebook. I switched fully to [yfinance](https://pypi.org/project/yfinance/) for working with Yahoo-Finance data due to its better maintenance and frequent new releases.

## How to run locally
If you want to run the notebooks locally on your laptop, clone the repo and `cd` into its directory, e.g.:
```commandline
git clone -l -s https://github.com/ilchen/US_Economic_Data_Analysis.git
cd US_Economic_Data_Analysis
```
run one of the below commands depending on which notebook you are interested in:
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
jupyter notebook Quantitative_Easing_and_Tapering.ipynb
```
or
```commandline
jupyter notebook Interest_Rate_Spreads.ipynb
```
or
```commandline
jupyter notebook Current_Riskfree_Rates.ipynb
```
or
```commandline
jupyter notebook Industrial_Production.ipynb
```
or
```commandline
jupyter notebook GDP_Composition.ipynb
```
or
```commandline
jupyter notebook Labor_Productivity.ipynb
```
or
```commandline
jupyter notebook Disposable_Income_and_Savings.ipynb
```
or
```commandline
jupyter notebook Stock_Market.ipynb
```

## How to run in Google cloud
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
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Quantitative_Easing_and_Tapering.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Industrial_Production.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/GDP_Composition.ipynb`
or `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Labor_Productivity.ipynb`
5. Click the search icon
6. Enjoy  
  In some of the notebooks I make use of additional python code I developed (e.g. `Current_Riskfree_Rates.ipynb`) or dependencies that are not by default provisioned in Google Colaboratory. When running these notebooks in Colaboratory, it's important to clone this repository and `cd` to it. I crated a commented out cell at the beginning of these notebooks to make it easier. Please don't forget to uncomment its content and run it first. E.g. here's one from `Current_Riskfree_Rates.ipynb`:
  ```commandline
# Uncomment if running in Google Colaboratory, otherwise the import of the curves module in the cell below will fail
#!git clone -l -s https://github.com/ilchen/US_Economic_Data_Analysis.git cloned-repo
#%cd cloned-repo

# Install the latest version of pandas-datareader and yfinance
# !pip install pandas-datareader -U
# !pip install yfinance -U
  ```
