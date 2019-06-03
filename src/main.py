from config import Config
import argparse


if __name__ == '__main__':
    # Load the configuration file
    config = Config()

    # Get the path and form of a construction from command-line
    parser = argparse.ArgumentParser(description="Automatic Annotator for Chinese construction corpora")
    parser.add_argument("-p", "--path", help="The path (specifically file name) of the raw material of the construction")
    parser.add_argument("-f", "--form", help="The abstract form of the construction")
    args = parser.parse_args()

