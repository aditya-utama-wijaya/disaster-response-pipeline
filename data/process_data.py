import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories datasets and merge them.
    
    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.
    
    Returns:
        df (pandas.DataFrame): Merged dataset containing messages and categories data.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on = "id", how = "inner")
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()
    category_colnames = [c for sublist in category_colnames for c in sublist]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    return df


def clean_data(df):
    """
    Clean the merged dataset by removing duplicates.
    
    Args:
        df (pandas.DataFrame): Merged dataset.
    
    Returns:
        df (pandas.DataFrame): Cleaned dataset without duplicates.
    """
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True)

    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset to a SQLite database.
    
    Args:
        df (pandas.DataFrame): Cleaned dataset.
        database_filename (str): Filename of the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index = False, if_exists = "replace")


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