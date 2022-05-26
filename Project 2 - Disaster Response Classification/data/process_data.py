import sys
from xml.dom.pulldom import IGNORABLE_WHITESPACE
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    loads:
    The specified messages and categories csv data files

    Args:
    messages_filepath(string): path of the messages data file
    categories_filepath(string): path of the categories data file

    Returns:
    pandas dataframe(df) that are merged on ID column
    """
    # import data from csv
    messages = pd.read_csv(messages_filepath)
    global categories
    categories = pd.read_csv(categories_filepath)

    # merge on ID
    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    Method to clean the data:
        - split categories into columns
        - check for non-binary values in columns and replace with appropriate value (0 or 1)
        - replace "categories" column from the new dataframe
        - concatenate with df and create a new dataframe
        - drop duplicates

    Args:
    df: input load dataframe

    Returns:
    Cleaned dataframe as final dataset for further processing

    """
    # create a dataframe with the 36 individual categories split at the ';' character
    categories = df.categories.str.split(';', expand=True)

    # extract first row and create columns; rename the column headers with the column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(
            lambda x: x[-1]).astype('str')
        categories[column] = categories[column].astype('int')

    # replace categories column in df with new category columns and concat original df with new dataframe
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop rows in "related column" with values other than 0 or 1
    df = df[df.related != 2]

    # drop columns which have only one class - no other value other than 0 (not classified)
    df.drop(columns='child_alone', inplace=True)

    return df


def save_data(df, database_filename, table_name):
    """ 
    Method to save final cleaned dataset into a sqlite database object

    Args:
        df_clean: cleaned dataset
        database_filename (str): name of the database to be created
        table_name (str): name of the table to create in the database

    Returns:
        None
    """

    # create engine
    engine = create_engine('sqlite:///' + database_filename)

    # load cleaned dataset into database, replacing it if it already exists
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        table_name = 'labeledmessages'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

