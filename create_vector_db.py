from rag import *
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Create Vector Data Base')
    parser.add_argument('-m', '--model', required=True, default='fnando1995/t5-small-ft-bookSum', type=str, help='Model to use from HF.')
    parser.add_argument('-d', '--directory', required=True, default = 'data',type=str, help='Directory where de datasets are.')
    parser.add_argument('-v', '--version', required=True, type=str, help='Version of the dataset to use. Normally preprocessed')
    return parser.parse_args()

def main():
    args = get_arguments()
    print("arguments created")
    model_name = args.model
    directory = args.directory
    version = args.version

    print("Creating RAG for Vector Database creation.")
    RAGT5(
        model_name
        , directory
        , version
        , '' # Not using now.
        , load_db_from_disk=False
    )


if __name__ == '__main__':
    main()