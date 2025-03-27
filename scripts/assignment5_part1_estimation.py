import sys, os, warnings

import gzip
import time

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

import utils

warnings.filterwarnings("ignore")

def main():
    unzipping_time_start = time.time()
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
    unzipping_time_end = time.time()
    print(f"\nUnzipping time: {unzipping_time_end - unzipping_time_start:.6f} seconds\n")

    appending_time_start = time.time()
    dataset = []
    for l in input_file:
        d = eval(l)
        dataset.append(d)
    input_file.close()
    appending_time_end = time.time()
    print(f"Appending time: {appending_time_end - appending_time_start:.6f} seconds\n")

    # code to split the data
    splitting_time_start = time.time()
    train_data = dataset[:int(len(dataset)*0.8)]
    dev_data = dataset[int(len(dataset)*0.8):]
    splitting_time_end = time.time()
    print(f"Splitting time: {splitting_time_end - splitting_time_start:.6f} seconds\n")


if __name__ == "__main__":
    main()

