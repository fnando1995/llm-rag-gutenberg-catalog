import os
import glob
import re
import argparse

def preprocess_text(text, split_string):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters (except for basic punctuation and spaces)
    text = re.sub(r'[^A-Za-z0-9\s,.!?\'\"]', '', text)
    # Remove extra spaces   
    text = ' '.join(text.split())
    # to lower
    text = text.lower()
    # every ebook have a preface divided by the text 'split_string'.
    text = text.split(split_string)[1]
    return text

def preprocess_data(fulldir,processes_full_dir,split_string):
    for ebook_path in glob.glob(os.path.join(fulldir,'*','*.txt')):
        with open(ebook_path,encoding='utf-8') as f:
            text = f.read()
        p_text = preprocess_text(text,split_string)
        new_path = ebook_path.replace(fulldir,processes_full_dir)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        with open(new_path,'w',encoding='utf-8') as f:
            f.write(p_text)

def get_arguments():
    parser = argparse.ArgumentParser(description='Preprocess PGC dataset')
    parser.add_argument('-d', '--directory', required=True, type=str, help='Main directory for the dataset.')
    parser.add_argument('-v', '--version', required=True, type=str, help='version of the downloaded dataset (%Y%m%d%H%M%S)')
    return parser.parse_args()

def main():
    args = get_arguments()
    directory       = args.directory
    version         = args.version
    split_string    = 'START OF THE PROJECT GUTENBERG EBOOK'.lower()
    fulldir = os.path.join(directory,version)
    processes_full_dir = fulldir + '_processed'
    os.makedirs(processes_full_dir, exist_ok=True)
    preprocess_data(fulldir,processes_full_dir,split_string)

    
if __name__ == '__main__':
    main()