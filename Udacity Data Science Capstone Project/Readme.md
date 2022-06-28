# Classification of Corporate Credit Ratings 

This repository contains the results of a data analysis performed on a set of corporate credit ratings given by ratings agencies to a set of companies. The aim of the data analysis is to build a machine learning model from the rating data that can be used to predict the rating a company will receive.

## Dataset
The dataset was generated with the file GenerateDataset.py. It makes use of an original dataset from Kaggle and Web Scraping exercise to combine this with a Standard Industry Classification table in Wikipedia. Reference links are provided in the acknowledgement section.

There are 25 features for every company of which 16 are financial indicators. They can be divided in:

Liquidity Measurement Ratios: CurrentRatio 
Profitability Indicator Ratios: EBIT margin, EBIDTA margin, GrossMargin, OperatingMargin, PretaxProfitMargin, NetProfitMargin, ReturnOnAssets, ReturnOnEquity, ReturnOnTangibleEquity,ReturnOnInvestment
Debt Ratios: LongtermDebt/Capital, DebtEquityRatio
Operating Performance Ratios: AssetTurnover
Cash Flow Indicator Ratios: OperatingCashFlowPerShare, freeCashFlowPerShare

## Results
Achieved an accuracy of 78.15 with an XGboost model post hyperparameter tuning.
![ML models](https://user-images.githubusercontent.com/28513435/176134473-999ffc8b-d97a-4b79-b168-2deff9d4a4c4.png)

## Conclusion
Next steps should be:
1. Employ data balancing techniques to see if we can achieve a higher accuracy
2. Build a web app and host it on the cloud

## Acknowledgments
Original Datasource: <a href=https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios>Kaggle</a>



