import urllib
import csv
from matplotlib.pyplot import table
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sphinx import RemovedInNextVersionWarning
from sympy import beta

# Read the original dataset on Corporate Credit Ratings & Financial Ratios
df_rating = pd.read_csv('corporateCreditRatingWithFinancialRatios.csv')
df_rating

# The "sector" mapping in this dataset is based on the French-Fama method.
# It maps a large number of rows to "Other". I wanted to have a more defined set of industry sectors, even if it was at a high level.
# So, I scraped the Industrial Classification table off the Wikipedia page as given in the url below and used that to provide a high level mapping of the industry sector for each SIC code

# Web Scraping the Wikipedia page to get the Industrial Classification table and use that to map sectors for each SIC Code

# Create  an URL object
url = 'https://en.wikipedia.org/wiki/Standard_Industrial_Classification'

# Create object page
page = requests.get(url)
page

# parser-lxml = Change html to Python friendly format
# Obtain page's information
soup = BeautifulSoup(page.text, 'lxml')
soup

# Obtain information from table class
table = soup.find("table", {"class": "wikitable sortable"})
table

# Obtain title of columns under the tag <th>
headers = []  # store headers in a list
for i in table.find_all('th'):
    title = i.text
    headers.append(title)

# print(headers.decode())

# Create a dataframe
SICRange = pd.DataFrame(columns=headers)
SICRange

# Fill data into the dataframe from the table under the tag <tr> under <td>
for j in table.find_all('tr')[1:]:
    row_data = j.find_all('td')
    row = [i.text for i in row_data]
    length = len(SICRange)
    SICRange.loc[length] = row

# Export to csv and read csv to create a dataframe for further processing
SICRange.to_csv('SICRangedata.csv', index=False)

# Read the SIC Codes data scraped from the Wikipedia site on Industry Classification
SICRange_data = pd.read_csv('SICRangedata.csv')
SICRange_data

# Clean the extra characters and change column names in the SICRange_data dataframe
SICRange_data.columns = SICRange_data.columns.str.replace('\n', '')
SICRange_data.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[
                      "", ""], regex=True, inplace=True)
SICRange_data.rename(
    columns={'Range of SIC Codes': 'CodeRange', 'Division': 'Sector'}, inplace=True)

# Split CodeRange column to Start & End columns + convert them to "int" type to enable comparison
SICRange_data[['Start', 'End']] = SICRange_data.CodeRange.apply(
    lambda x: pd.Series(str(x).split("-")))
SICRange_data['End'] = pd.to_numeric(SICRange_data['End'])
SICRange_data['Start'] = pd.to_numeric(SICRange_data['Start'])

# Check for each SIC code to be within the range ("Start" & "End") and map industry sector accordingly in original dataframe
data = []
for i in range(len(df_rating)):
    code = df_rating['SIC Code'].iloc[i]
    for j in range(len(SICRange_data)):
        start = SICRange_data['Start'].iloc[j]
        end = SICRange_data['End'].iloc[j]
        if code >= start and code <= end:
            data.append(SICRange_data['Sector'].iloc[j])
            continue
df_rating['Sector'] = data
df_rating

# Generate dataset from modified dataframe for the actual analysis and ML
df_rating.to_csv(
    'Corporate_Credit_Ratings.csv', index=False)
