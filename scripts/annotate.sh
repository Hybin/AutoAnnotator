#!/bin/bash

# transfer into the src directory
cd ../src

corpora=../data/input/*

for file in ${corpora}
do
    # Get the file name with .xml format
    corpus="$(cut -d'/' -f4 <<< ${file})"
    echo "Done! Get the corpus file - ["${corpus}"]"

    # Get the file name without .xml format
    filename="$(cut -d'.' -f1 <<< ${corpus})"

    # Get the form of the construction
    form="$(cut -d'_' -f1 <<< ${filename})"
    echo "Done! Get the pattern of construction - ["${form}"]"

    # Run python scripts
    # Standard Mode
    python3 main.py --path=${corpus} --form=${form} --mode=standard
    # Pipeline Mode
    python3 main.py --path=${corpus} --form=${form} --mode=pipeline
done



