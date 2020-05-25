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
    """
    This function aims to clean the data by removing rows with NaN in 'message' and 'categories' column and
    duplicate disaster messages.
    This function splits the single column with 36 categories to 36 columns and the value of each category column
    will be extracted.

    Args:
        df (df) -- a pandas dataframe that consists of disaster messages and corresponding categories,
        returned from load_data()

    Return:
        df_cleaned (df) -- a clean pandas dataframe with the newly added 36 disaster category columns

    """

    # Drop rows that do not have valid data in messages or categories col
    if (df['message'].isna().sum() != 0) | (df['categories'].isna().sum() != 0):
        df[['message', 'categories']].dropna(inplace=True)
        print("    NaN in 'message' and 'categories' column found. Dropping NaN rows...")

    # Create a dataframe of the 36 individual category columns
    categories = pd.Series(df['categories']).str.split(pat=';', expand=True)

    # Use one row of the categories dataframe to extract a list of new column names for categories
    row = categories.loc[1, :]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of categories
    categories.columns = category_colnames

    # Convert category values to numbers, just 0 or 1 which is the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # Drop the original 'categories' column from original dataframe, df
    df.drop(columns=['categories'], axis=1, inplace=True)

    # Concatenate the original dataframe, df with the new categories dataframe
    df_cleaned = pd.concat([df, categories], axis=1)

    # Remove duplicates (id) in the dataset
    if df_cleaned.duplicated(subset='id').sum() != 0:
        df_cleaned.drop_duplicates(subset='id', inplace=True)
        print("    Duplicate disaster messages found in dataset. Deleting duplicates...")

    return df_cleaned


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