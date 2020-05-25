import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads the csv files of the disaster messages and disaster categories.
    It will then merge them together to form a single dataset.

    Note:
        Make sure column 'id' exists in both datasets so that they can be merged.

    Args:
        messages_filepath (str) -- csv filepath of disaster messages
        categories_filepath (str) -- csv filepath of disaster categories

    Return:
        df_merged (df) -- a pandas dataframe consists of the disaster messages and corresponding categories

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df_merged = pd.merge(messages, categories, on='id')

    return df_merged


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()