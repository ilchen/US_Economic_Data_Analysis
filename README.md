# US Economic Data Analysis
This repository contains Jupyter notebooks that visually analyze US Economic data as provided by [St. Louis Fed](https://fred.stlouisfed.org).
The analysis is carried out using [Pandas](https://pandas.pydata.org), [Pandas datareader](https://pydata.github.io/pandas-datareader/), and [Matplotlib](https://matplotlib.org/stable/index.html).

So far I created the following notebooks:
* [Analysis of evolution of the seasonally adjusted CPI, Fed Funds Effective Rate, and Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity](./CPI_and_Fed_Funds_Rates.ipynb)
* [Analysis of evolution of the US Federal Public Debt as percentage of GDP, Public Debt as percentage of Federal annual tax revenue, and Annual pulic deficit as percentage of GDP](./Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb)

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything and run these notebooks in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```

## How to run
After you clone the repo and `cd` into its directory:
```commandline
jupyter notebook CPI_and_Fed_Funds_Rates.ipynb
```
or
```commandline
jupyter notebook Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb
```
A full run of these notebooks can be seen [here for CPI](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/CPI_and_Fed_Funds_Rates.ipynb) and
[here for public debt analysis](https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb).  

You can also run these notebooks in Google cloud. This way you don't need to install anything locally. This takes just a few seconds:
1. Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in your browser
2. In the modal window that appears select `GitHub`
3. Enter the URL of this repository's notebook, e.g.: `https://github.com/ilchen/US_Economic_Data_Analysis/blob/main/Fed_Public_Debt_and_Fed_Tax_Revenue.ipynb`
4. Click the search icon
5. Enjoy
