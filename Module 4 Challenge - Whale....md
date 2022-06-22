```python
from pathlib import Path
import pandas as pd
import csv
import seaborn as sns
import datetime as dt
import numpy as np
%matplotlib inline

```


```python
#Prepare the Data
```


```python
whale_returns_pth = Path('/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/Starter_Code/Resources/whale_returns.csv')
algo_returns_pth = Path('/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/Starter_Code/Resources/algo_returns.csv')
sp_tsx_history_pth = Path('/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/Starter_Code/Resources/sp_tsx_history.csv')
```


```python
whale_returns_df = pd.read_csv(whale_returns_pth, index_col="Date", infer_datetime_format=True, parse_dates=True)
whale_returns_df.sort_index(ascending = True, inplace = True)

algo_returns_df = pd.read_csv(algo_returns_pth, index_col="Date", infer_datetime_format=True, parse_dates=True)
algo_returns_df.sort_index(ascending = True, inplace = True)

sp_tsx_history_df = pd.read_csv(sp_tsx_history_pth, index_col="Date", infer_datetime_format=True, parse_dates=True)
sp_tsx_history_df.sort_index(ascending = True, inplace = True)
sp_tsx_history_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-01</th>
      <td>$12,370.19</td>
    </tr>
    <tr>
      <th>2012-10-02</th>
      <td>$12,391.23</td>
    </tr>
    <tr>
      <th>2012-10-03</th>
      <td>$12,359.47</td>
    </tr>
    <tr>
      <th>2012-10-04</th>
      <td>$12,447.68</td>
    </tr>
    <tr>
      <th>2012-10-05</th>
      <td>$12,418.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
whale_returns_df[whale_returns_df.isnull().any(axis = 1)]
whale_returns_df.dropna(inplace=True)
whale_returns_df.isnull().sum()

```




    SOROS FUND MANAGEMENT LLC      0
    PAULSON & CO.INC.              0
    TIGER GLOBAL MANAGEMENT LLC    0
    BERKSHIRE HATHAWAY INC         0
    dtype: int64




```python
algo_returns_df.isnull().sum()
algo_returns_df.dropna(inplace=True)
algo_returns_df.isnull().sum()
```




    Algo 1    0
    Algo 2    0
    dtype: int64




```python
sp_tsx_history_df.dtypes
```




    Close    object
    dtype: object




```python
sp_tsx_history_df["Close"]= sp_tsx_history_df["Close"].str.replace("$","").str.replace(',','').astype("float")
sp_tsx_history_df.head()
```

    /var/folders/n4/gpv7p5n93k529lgh_r2dcw4h0000gn/T/ipykernel_36692/2113737378.py:1: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      sp_tsx_history_df["Close"]= sp_tsx_history_df["Close"].str.replace("$","").str.replace(',','').astype("float")





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-01</th>
      <td>12370.19</td>
    </tr>
    <tr>
      <th>2012-10-02</th>
      <td>12391.23</td>
    </tr>
    <tr>
      <th>2012-10-03</th>
      <td>12359.47</td>
    </tr>
    <tr>
      <th>2012-10-04</th>
      <td>12447.68</td>
    </tr>
    <tr>
      <th>2012-10-05</th>
      <td>12418.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
sp_tsx_daily_returns = sp_tsx_history_df.pct_change()

sp_tsx_daily_returns.dropna(inplace = True)
sp_tsx_daily_returns.head()

sp_tsx_daily_returns.rename(columns = {"Close":"S&P_TSX"}, inplace = True)
```


```python
combined_df = pd.concat([whale_returns_df, algo_returns_df,sp_tsx_daily_returns], axis = "columns",join = "inner")
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P_TSX</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-03</th>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
      <td>-0.001942</td>
      <td>-0.000949</td>
      <td>-0.008530</td>
    </tr>
    <tr>
      <th>2015-03-04</th>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
      <td>-0.008589</td>
      <td>0.002416</td>
      <td>-0.003371</td>
    </tr>
    <tr>
      <th>2015-03-05</th>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
      <td>-0.000955</td>
      <td>0.004323</td>
      <td>0.001344</td>
    </tr>
    <tr>
      <th>2015-03-06</th>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
      <td>-0.004957</td>
      <td>-0.011460</td>
      <td>-0.009972</td>
    </tr>
    <tr>
      <th>2015-03-09</th>
      <td>0.000582</td>
      <td>0.004225</td>
      <td>0.005843</td>
      <td>-0.001652</td>
      <td>-0.005447</td>
      <td>0.001303</td>
      <td>-0.006555</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Analysis
```


```python
combined_df.plot(figsize = (20,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_11_1.png)
    



```python
Cumulative_df = (1+ combined_df).cumprod()
Cumulative_df.plot(figsize = (20,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_12_1.png)
    



```python
#Risk Analysis 
combined_df.plot(kind = "box", figsize = (20,10))
```




    <AxesSubplot:>




    
![png](output_13_1.png)
    



```python
std_daily_df = pd.DataFrame(combined_df.std()).rename(columns = {0:"std"})
std_daily_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <td>0.007828</td>
    </tr>
    <tr>
      <th>PAULSON &amp; CO.INC.</th>
      <td>0.006982</td>
    </tr>
    <tr>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <td>0.010883</td>
    </tr>
    <tr>
      <th>BERKSHIRE HATHAWAY INC</th>
      <td>0.012826</td>
    </tr>
    <tr>
      <th>Algo 1</th>
      <td>0.007589</td>
    </tr>
    <tr>
      <th>Algo 2</th>
      <td>0.008326</td>
    </tr>
    <tr>
      <th>S&amp;P_TSX</th>
      <td>0.007034</td>
    </tr>
  </tbody>
</table>
</div>




```python
std_higher_than_sp_tsx = std_daily_df[std_daily_df["std"] > std_daily_df.loc["S&P_TSX", "std"]]
std_higher_than_sp_tsx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <td>0.007828</td>
    </tr>
    <tr>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <td>0.010883</td>
    </tr>
    <tr>
      <th>BERKSHIRE HATHAWAY INC</th>
      <td>0.012826</td>
    </tr>
    <tr>
      <th>Algo 1</th>
      <td>0.007589</td>
    </tr>
    <tr>
      <th>Algo 2</th>
      <td>0.008326</td>
    </tr>
  </tbody>
</table>
</div>




```python
annualized_std_df = std_daily_df*np.sqrt(252)
annualized_std_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <td>0.124259</td>
    </tr>
    <tr>
      <th>PAULSON &amp; CO.INC.</th>
      <td>0.110841</td>
    </tr>
    <tr>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <td>0.172759</td>
    </tr>
    <tr>
      <th>BERKSHIRE HATHAWAY INC</th>
      <td>0.203599</td>
    </tr>
    <tr>
      <th>Algo 1</th>
      <td>0.120470</td>
    </tr>
    <tr>
      <th>Algo 2</th>
      <td>0.132177</td>
    </tr>
    <tr>
      <th>S&amp;P_TSX</th>
      <td>0.111664</td>
    </tr>
  </tbody>
</table>
</div>




```python
#ROLLING STATISTICS
sp_tsx_rolling = combined_df[["S&P_TSX"]].rolling(window = 21).std()
sp_tsx_rolling.plot(figsize = (20,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_17_1.png)
    



```python
correlation_sp_tsx = pd.DataFrame(combined_df.corr()).loc[:,"S&P_TSX"]
correlation_sp_tsx[correlation_sp_tsx==correlation_sp_tsx[correlation_sp_tsx<1].max()]
```




    Algo 2    0.73737
    Name: S&P_TSX, dtype: float64




```python
combined_df.plot(kind="scatter", y = "SOROS FUND MANAGEMENT LLC", x = "S&P_TSX", figsize = (20,10))
combined_df.plot(kind="scatter", y = "Algo 2", x = "S&P_TSX", figsize = (20,10))

```




    <AxesSubplot:xlabel='S&P_TSX', ylabel='Algo 2'>




    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    



```python
covariance_rolling= combined_df['TIGER GLOBAL MANAGEMENT LLC'].rolling(window=60).cov(combined_df['S&P_TSX'])
variance_rolling = combined_df['S&P_TSX'].rolling(window=60).var()
beta_rolling = covariance_rolling / variance_rolling
beta_rolling.plot(figsize=(20, 10), title='Rolling 60-Day Beta of TIGER GLOBAL MANAGEMENT')
```




    <AxesSubplot:title={'center':'Rolling 60-Day Beta of TIGER GLOBAL MANAGEMENT'}, xlabel='Date'>




    
![png](output_20_1.png)
    



```python
#Sharpe Ratios
sharpe_ratios = combined_df.mean()*252/(combined_df.std()*np.sqrt(252))
sharpe_ratios
```




    SOROS FUND MANAGEMENT LLC      0.286709
    PAULSON & CO.INC.             -0.547594
    TIGER GLOBAL MANAGEMENT LLC   -0.144455
    BERKSHIRE HATHAWAY INC         0.467045
    Algo 1                         1.491514
    Algo 2                         0.396817
    S&P_TSX                        0.195550
    dtype: float64




```python
sharpe_ratios.plot(kind = "bar", title = "Sharpe Ratios for Portfolios")
```




    <AxesSubplot:title={'center':'Sharpe Ratios for Portfolios'}>




    
![png](output_22_1.png)
    



```python
#The Algo 1 outperformed both the market and the respective whales, whereas Algo 2 outperformed the
#market and all whales except berkshire hathaway
```


```python
#Custom Portfolio
```


```python
AAPL_df= pd.read_csv(Path("/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/AAPL DATA.csv"), index_col="Date", 
                           parse_dates = True, infer_datetime_format= True)
AAPL_df.rename(columns = {"Close":"AAPL"}, inplace = True)
AAPL_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-02 16:00:00</th>
      <td>32.27</td>
    </tr>
    <tr>
      <th>2015-03-03 16:00:00</th>
      <td>32.34</td>
    </tr>
    <tr>
      <th>2015-03-04 16:00:00</th>
      <td>32.14</td>
    </tr>
    <tr>
      <th>2015-03-05 16:00:00</th>
      <td>31.60</td>
    </tr>
    <tr>
      <th>2015-03-06 16:00:00</th>
      <td>31.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
MSFT_df= pd.read_csv(Path("/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/MSFT DATA.csv"), index_col="Date", 
                           parse_dates = True, infer_datetime_format= True)
MSFT_df.rename(columns = {"Close":"MSFT"}, inplace = True)
MSFT_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSFT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-02 16:00:00</th>
      <td>43.88</td>
    </tr>
    <tr>
      <th>2015-03-03 16:00:00</th>
      <td>43.28</td>
    </tr>
    <tr>
      <th>2015-03-04 16:00:00</th>
      <td>43.06</td>
    </tr>
    <tr>
      <th>2015-03-05 16:00:00</th>
      <td>43.11</td>
    </tr>
    <tr>
      <th>2015-03-06 16:00:00</th>
      <td>42.36</td>
    </tr>
  </tbody>
</table>
</div>




```python
NFLX_df= pd.read_csv(Path("/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/NFLX DATA.csv"), index_col="Date", 
                           parse_dates = True, infer_datetime_format= True)
NFLX_df.rename(columns = {"Close":"NFLX"}, inplace = True)
NFLX_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NFLX</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-02 16:00:00</th>
      <td>68.61</td>
    </tr>
    <tr>
      <th>2015-03-03 16:00:00</th>
      <td>67.82</td>
    </tr>
    <tr>
      <th>2015-03-04 16:00:00</th>
      <td>67.11</td>
    </tr>
    <tr>
      <th>2015-03-05 16:00:00</th>
      <td>66.81</td>
    </tr>
    <tr>
      <th>2015-03-06 16:00:00</th>
      <td>64.87</td>
    </tr>
  </tbody>
</table>
</div>




```python
SHOP_df= pd.read_csv(Path("/Users/ameerirfan/Desktop/FinTech_Bootcamp/Challenges/Module 4/SHOP DATA.csv"), index_col="Date", 
                           parse_dates = True, infer_datetime_format= True)
SHOP_df.rename(columns = {"Close":"SHOP"}, inplace = True)
SHOP_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SHOP</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-05-21 16:00:00</th>
      <td>25.68</td>
    </tr>
    <tr>
      <th>2015-05-22 16:00:00</th>
      <td>28.31</td>
    </tr>
    <tr>
      <th>2015-05-26 16:00:00</th>
      <td>29.65</td>
    </tr>
    <tr>
      <th>2015-05-27 16:00:00</th>
      <td>27.50</td>
    </tr>
    <tr>
      <th>2015-05-28 16:00:00</th>
      <td>27.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
portfolio_stocks = pd.concat((AAPL_df, MSFT_df, NFLX_df, SHOP_df), axis = 1, join= "inner")
portfolio_stocks.sort_index(ascending = True, inplace = True)
portfolio_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>MSFT</th>
      <th>NFLX</th>
      <th>SHOP</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-05-21 16:00:00</th>
      <td>32.85</td>
      <td>47.42</td>
      <td>89.00</td>
      <td>25.68</td>
    </tr>
    <tr>
      <th>2015-05-22 16:00:00</th>
      <td>33.14</td>
      <td>46.90</td>
      <td>88.84</td>
      <td>28.31</td>
    </tr>
    <tr>
      <th>2015-05-26 16:00:00</th>
      <td>32.41</td>
      <td>46.59</td>
      <td>87.99</td>
      <td>29.65</td>
    </tr>
    <tr>
      <th>2015-05-27 16:00:00</th>
      <td>33.01</td>
      <td>47.61</td>
      <td>89.86</td>
      <td>27.50</td>
    </tr>
    <tr>
      <th>2015-05-28 16:00:00</th>
      <td>32.95</td>
      <td>47.45</td>
      <td>89.51</td>
      <td>27.45</td>
    </tr>
  </tbody>
</table>
</div>




```python
portfolio_stocks.dropna(inplace=True)
```


```python
weights = [1/4, 1/4, 1/4,1/4]
portfolio_total = portfolio_stocks.dot(weights)
portfolio_returns = portfolio_total.pct_change()
portfolio_returns.head()
```




    Date
    2015-05-21 16:00:00         NaN
    2015-05-22 16:00:00    0.011490
    2015-05-26 16:00:00   -0.002789
    2015-05-27 16:00:00    0.006814
    2015-05-28 16:00:00   -0.003132
    dtype: float64




```python

Total_portfolio_returns = pd.concat([portfolio_returns, combined_df], axis = "columns",join = "outer")
Total_portfolio_returns.rename(columns = {0:"MY PORTFOLIO"}, inplace = True)
Total_portfolio_returns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MY PORTFOLIO</th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P_TSX</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-03 00:00:00</th>
      <td>NaN</td>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
      <td>-0.001942</td>
      <td>-0.000949</td>
      <td>-0.008530</td>
    </tr>
    <tr>
      <th>2015-03-04 00:00:00</th>
      <td>NaN</td>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
      <td>-0.008589</td>
      <td>0.002416</td>
      <td>-0.003371</td>
    </tr>
    <tr>
      <th>2015-03-05 00:00:00</th>
      <td>NaN</td>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
      <td>-0.000955</td>
      <td>0.004323</td>
      <td>0.001344</td>
    </tr>
    <tr>
      <th>2015-03-06 00:00:00</th>
      <td>NaN</td>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
      <td>-0.004957</td>
      <td>-0.011460</td>
      <td>-0.009972</td>
    </tr>
    <tr>
      <th>2015-03-09 00:00:00</th>
      <td>NaN</td>
      <td>0.000582</td>
      <td>0.004225</td>
      <td>0.005843</td>
      <td>-0.001652</td>
      <td>-0.005447</td>
      <td>0.001303</td>
      <td>-0.006555</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-04-29 00:00:00</th>
      <td>NaN</td>
      <td>0.001254</td>
      <td>0.002719</td>
      <td>0.006251</td>
      <td>0.005223</td>
      <td>0.005208</td>
      <td>0.002829</td>
      <td>-0.000788</td>
    </tr>
    <tr>
      <th>2019-04-29 16:00:00</th>
      <td>0.000565</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-04-30 00:00:00</th>
      <td>NaN</td>
      <td>-0.001295</td>
      <td>-0.002211</td>
      <td>-0.000259</td>
      <td>-0.003702</td>
      <td>-0.002944</td>
      <td>-0.001570</td>
      <td>-0.001183</td>
    </tr>
    <tr>
      <th>2019-04-30 16:00:00</th>
      <td>0.020937</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-05-01 00:00:00</th>
      <td>NaN</td>
      <td>-0.005847</td>
      <td>-0.001341</td>
      <td>-0.007936</td>
      <td>-0.007833</td>
      <td>0.000094</td>
      <td>-0.007358</td>
      <td>-0.004703</td>
    </tr>
  </tbody>
</table>
<p>2021 rows × 8 columns</p>
</div>




```python
Total_portfolio_returns.dropna(inplace=True)
Total_portfolio_returns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MY PORTFOLIO</th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>S&amp;P_TSX</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python

```
