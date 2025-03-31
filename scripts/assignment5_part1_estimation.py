import sys, os, warnings

import gzip
import time
import pickle
import numpy as np

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer # type: ignore

import utils # type: ignore
import utils

warnings.filterwarnings("ignore")

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
    # DEBUGGING Print sizes of train_data and dev_data in bytes
    # print(f"Train data size: {sum(len(pickle.dumps(d)) for d in train_data)} bytes")
    # print(f"Dev data size: {sum(len(pickle.dumps(d)) for d in dev_data)} bytes\n")

    # TODO: This should be a function inside utils that takes in data and a string that
    # corresponds to a json feature and it returns a vector of just the features of data

    train_texts = train_data['text'].tolist()
    train_hours = train_data['hours'].values.reshape(-1, 1)

    # discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    # y_hours_binned = discretizer.fit_transform(train_hours).ravel().astype(int)
    
    y_hours_binned = np.digitize(train_hours, [2, 4, 10, 50, 100, 500, 1000], right=False)

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
    NB_pipeline.fit(train_texts, y_hours_binned)
    NB_time_end = time.time()
    print(f"Training time: {NB_time_end - NB_time_start:.6f} seconds\n")

    dev_texts = dev_data['text'].tolist()
    dev_hours = dev_data['hours'].values.reshape(-1, 1)
    
    # y_dev_hours_binned = discretizer.transform(dev_hours).ravel().astype(int)
    y_dev_hours_binned = np.digitize(dev_hours, [2, 4, 10, 50, 100, 500, 1000], right=False)
    y_pred = NB_pipeline.predict(dev_texts)
    # 4) Evaluate accuracy
    dev_accuracy = accuracy_score(y_dev_hours_binned, y_pred)
    print(f"Accuracy on dev set: {100*dev_accuracy:.1f}%")




if __name__ == "__main__":
    main()