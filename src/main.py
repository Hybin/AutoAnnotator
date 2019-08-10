from config import Config
from annotator import Annotator
from pipeline import Pipeline
import argparse


if __name__ == '__main__':
    # Load the configuration file
    config = Config()

    # Get the path and form of a construction from command-line
    parser = argparse.ArgumentParser(description="Automatic Annotator for Chinese construction corpora")
    parser.add_argument("-p", "--path", help="The path (specifically file name) of the raw material of the construction")
    parser.add_argument("-f", "--form", help="The abstract form of the construction")
    parser.add_argument("-m", "--mode", help="The mode of the system, the value could be one of [standard, pipeline]")
    args = parser.parse_args()

    if args.mode == "standard":
        # Begin to Annotate
        annotator = Annotator(config, args.form, args.path)
        annotator.initialize()
        annotator.store()
    else:
        pipeline = Pipeline(config, args.form, args.path)
        pipeline.annotate()
