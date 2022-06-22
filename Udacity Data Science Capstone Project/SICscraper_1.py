import urllib
import csv
from matplotlib.pyplot import table
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sphinx import RemovedInNextVersionWarning
from sympy import beta

# create  an URL object

url = 'https://en.wikipedia.org/wiki/Standard_Industrial_Classification'

# create object page
page = requests.get(url)
page

# parser-lxml = Change html to Python friendly format
# Obtain page's information
soup = BeautifulSoup(page.text, 'lxml')
soup

# obtain information from table class

table = soup.find("table", {"class": "wikitable sortable"})
table

# obtain title of columns under the tag <th>
headers = []  # store headers in a list
for i in table.find_all('th'):
    title = i.text
    headers.append(title)

# print(headers.decode())

# create a dataframe
SICRange = pd.DataFrame(columns=headers)

# fill data into the dataframe from the table under the tag <tr> under <td>
for j in table.find_all('tr')[1:]:
    row_data = j.find_all('td')
    row = [i.text for i in row_data]
    length = len(SICRange)
    SICRange.loc[length] = row

# print(SICRange.decode())
# export to csv

SICRange.to_csv('SICRangedata.csv', index=False)

# read the csv
SIC_df = pd.read_csv('SICRangedata.csv')
# SIC_df.head()
