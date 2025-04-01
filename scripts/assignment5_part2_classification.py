import sys, os, warnings

import gzip
import time
import pickle
import numpy as np

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.preprocessing import KBinsDiscretizer # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

import utils
warnings.filterwarnings("ignore")

NUMBER_OF_BINS = 2
REMOVE_OUTLIERS = True
LOG_TRANSFORM = True

def log_transform(labels) -> np.ndarray:
    return np.log2(labels + 1)
    

def main():
    unzip_time_start = time.time()
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
    unzip_time_end = time.time()
    print(f"\nUnzipping time: {unzip_time_end - unzip_time_start:.6f} seconds\n")
    # DEBUGGING Print size of input file 
    # input_file.seek(0, 2)
    # print(f"Input file size: {input_file.tell()} bytes\n")
    # input_file.seek(0)

    appending_time_start = time.time()
    dataset = []
    for i, l in enumerate(input_file):
        # if i > 1000:
        #     break
        d = eval(l)
        dataset.append(d)
    input_file.close()
    appending_time_end = time.time()
    print(f"Appending time: {appending_time_end - appending_time_start:.6f} seconds\n")
    # DEBUGGING Print size of dataset in bytes
    # print(f"Dataset size: {sum(len(pickle.dumps(d)) for d in dataset)} bytes\n")

    cleaning_time_start = time.time()
    data = utils.clean_data(dataset, required_features=["hours", "early_access", "text"])
    cleaning_time_end = time.time()
    print(f"\Cleaning time: {cleaning_time_end - cleaning_time_start:.6f} seconds\n")


    # code to split the data
    splitting_time_start = time.time()
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    dev_data   = data[split_index:]
    splitting_time_end = time.time()

    print(f"Splitting time: {splitting_time_end - splitting_time_start:.6f} seconds\n")
    

if __name__ == "__main__":
    main()