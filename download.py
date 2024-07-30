import os
import requests
import pandas as pd
import math
import random
from datetime import datetime as dt
import argparse


def download(book_id, category, directory):
    '''
    Function to download one book. Saved ad directory/category/book_{book_id}.txt
    '''
    try:
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        os.makedirs(os.path.join(directory, category), exist_ok=True)
        filename = os.path.join(directory, category, f"book_{book_id}.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Sucess!: Category {category} - book_{book_id}.txt")
    except Exception as e:
        print(f"Fail!: Category {category} - book_{book_id}.txt")


def download_random(dataset_directory,n_categories,n_books):
    # download metadata of PGC 
    dfo = pd.read_csv("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",sep=",",dtype=str)
    dfo.to_csv(os.path.join(dataset_directory,'pg_catalog_original.csv'),index=False)

    # filter by language and category.
    def filternancat(row):
        if type(row['category'])==float and math.isnan(row['category']):
            return False
        return True
    df = dfo[(dfo['Language']=="en")]
    df = df[['Text#','Title','Language','Bookshelves']]
    df.rename({'Text#':'bookid','Title':'title','Language':'language','Bookshelves':'category'},axis=1,inplace=True)
    df = df[df.apply(filternancat, axis=1)]
    df['category'] = [catg.split(';')[0] for catg in df['category']] # get first category in Bookshelves(category) as the main category

    # generate new directory with datetime
    date_dataset = dt.now().strftime("%Y%m%d%H%M%S")
    datedir = os.path.join(dataset_directory,date_dataset)

    # select random categories and random book per categories to download.
    random_categories = random.choices(list(set(df['category'])), k=n_categories) 
    for category in random_categories:
        book_ids = random.choices(list(df[df['category']==category]['bookid']), k=n_books)
        for book_id in book_ids:
            download(book_id,category,datedir)

    print("All books downloaded!")

def get_arguments():
    parser = argparse.ArgumentParser(description='Download PGC dataset')
    parser.add_argument('-d', '--directory', required=True, type=str, help='Main directory for the dataset.')
    parser.add_argument('-c', '--categories_number', required=True, type=int, help='Number of random categories to download.')
    parser.add_argument('-b', '--book_number', required=True, type=int, help='Number of random books per category to download.')
    return parser.parse_args()

def main():
    args = get_arguments()
    directory = args.directory
    n_categories = args.categories_number
    n_books = args.book_number
    download_random(directory,n_categories,n_books)

if __name__ == '__main__':
    main()