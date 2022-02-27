NYU NLP Homework 4: Implement an ad hoc information retrieval system
    using TF-IDF weights and cosine similarity scores.
    by Ziyang Zeng (zz2960)
    Spring 2022

Pre-requisites:
    - Python 3.8+

Install dependencies:
    `pip3 install -r requirements.txt`

How to run:
    `python3 main_zz2960_HW4.py --help` will give you:
        usage: main_zz2960_HW4.py [-h] [-o OUTPUTFILE] articlefile queryfile

        An ad hoc information retrieval system using TF-IDF weights and cosine similarity scores.

        positional arguments:
        articlefile    input article corpus file
        queryfile      input query file

        optional arguments:
        -h, --help     show this help message and exit
        -o OUTPUTFILE  path for storing output file, default is output.txt

Example:
    `python3 main_zz2960_HW4.py data/cran.all.1400 data/cran.qry -o output.txt`
