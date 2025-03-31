import sys, os, warnings

import gzip
import time
import pickle

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

import utils

warnings.filterwarnings("ignore")

def main():
    unzipping_time_start = time.time()
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
    unzipping_time_end = time.time()
    print(f"\nUnzipping time: {unzipping_time_end - unzipping_time_start:.6f} seconds\n")
    # DEBUGGING Print size of input file 
    # input_file.seek(0, 2)
    # print(f"Input file size: {input_file.tell()} bytes\n")
    # input_file.seek(0)

    appending_time_start = time.time()
    dataset = []
    for l in input_file:
        d = eval(l)
        dataset.append(d)
    input_file.close()
    appending_time_end = time.time()
    print(f"Appending time: {appending_time_end - appending_time_start:.6f} seconds\n")
    # DEBUGGING Print size of dataset in bytes
    # print(f"Dataset size: {sum(len(pickle.dumps(d)) for d in dataset)} bytes\n")

    ## TODO: call utils.clean_data(dataset) to get a clean dataset
    ##       we still don't know what cleaning this data set will look like

    # code to split the data
    splitting_time_start = time.time()
    train_data = dataset[:int(len(dataset)*0.8)]
    dev_data = dataset[int(len(dataset)*0.8):]
    splitting_time_end = time.time()
    print(f"Splitting time: {splitting_time_end - splitting_time_start:.6f} seconds\n")
    # DEBUGGING Print sizes of train_data and dev_data in bytes
    # print(f"Train data size: {sum(len(pickle.dumps(d)) for d in train_data)} bytes")
    # print(f"Dev data size: {sum(len(pickle.dumps(d)) for d in dev_data)} bytes\n")

    # TODO: This should be a function inside utils that takes in data and a string that
    # corresponds to a json feature and it returns a vector of just the features of data
    print(train_data[0]['hours'])
    train_texts = [entry['text'] for entry in train_data]
    train_hours = [entry['hours'] for entry in train_data]



    #############################
    ## Multinomial Naive Bayes ##
    #############################
    # Make prediction based on Multinomial Naive Bayes
    NB_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            # There are more options for this
            # relevant ones are max_df, min_df, and max_features
            # for documentation read:
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
            strip_accents='ascii', 
            lowercase=True, 
            analyzer='word', 
            stop_words='english')),
        ('classifier', MultinomialNB()),
    ])
    print('\nPerforming Multinomial Naive Bayes on the reviews')
    NB_time_start = time.time()
    NB_pipeline.fit(train_texts, train_hours)
    NB_time_end = time.time()
    print(f"Splitting time: {NB_time_end - NB_time_start:.6f} seconds\n")



if __name__ == "__main__":
    main()