import sys, os, warnings

import gzip
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer, StandardScaler # type: ignore

import utils
warnings.filterwarnings("ignore")

NUMBER_OF_BINS = 2

def compute_text_length(X):
    return np.array([len(text) for text in X]).reshape(-1, 1)

def complete_classifier():
    # Load File
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")

    # Append Data
    appending_time_start = time.time()
    dataset = []
    for i, l in enumerate(input_file):
        if i > 1000:
            break
        d = eval(l)
        dataset.append(d)
    input_file.close()
    appending_time_end = time.time()
    print(f"Loading and Appending Data time: {appending_time_end - appending_time_start:.6f} seconds\n")


    # Clean Data
    cleaning_time_start = time.time()
    data = utils.clean_data(dataset, required_features=["hours", "early_access", "text"])
    cleaning_time_end = time.time()
    print(f"Cleaning time: {cleaning_time_end - cleaning_time_start:.6f} seconds\n")
    

    # Split Data
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    dev_data   = data[split_index:]


    # Extract Relevant Information From Data
    train_texts = train_data['text'].tolist()
    train_hours = train_data['hours'].values.reshape(-1, 1)

    dev_texts = dev_data['text'].tolist()
    dev_hours = dev_data['hours'].values.reshape(-1, 1)


    # Discretize the hours into quartiles (bins) of both training and dev datasets
    discretizer = KBinsDiscretizer(n_bins=NUMBER_OF_BINS, encode='ordinal', strategy='quantile')

    train_discretizer_time_start = time.time()
    train_hours_binned = discretizer.fit_transform(train_hours).ravel().astype(int)
    train_discretizer_time_end = time.time()
    print(f"Train labels discretizer transform time for {NUMBER_OF_BINS} bins: {train_discretizer_time_end - train_discretizer_time_start:.6f} seconds\n")

    dev_discretizer_time_start = time.time()
    dev_hours_binned = discretizer.transform(dev_hours).ravel().astype(int)
    dev_discretizer_time_end = time.time()
    print(f"Dev labels discretizer transform time: {dev_discretizer_time_end - dev_discretizer_time_start:.6f} seconds\n")


    ####################
    ## Classification ##
    ####################

    # Feature Processing Pipeline
    feature_transformer = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', stop_words='english'), 'text'),
        ('text_length', Pipeline([
            ('compute_length', FunctionTransformer(compute_text_length, validate=False))
        ]), 'text'),
        ('early_access', 'passthrough', ['early_access'])
    ])

    ## Classifier Number 1 - Multinomial Naive Bayes ##
    print('\nPerforming Multinomial Naive Bayes on the reviews')

    # Create MNB pipeline
    NB_pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', MultinomialNB()),
    ])

    # Train Classifier
    NB_time_start = time.time()
    NB_pipeline.fit(train_data, train_hours_binned)
    NB_time_end = time.time()
    print(f"\tNB train time: {NB_time_end - NB_time_start:.6f} seconds")

    # Predict Developer Data
    dev_data_predict_time_start = time.time()
    nb_dev_pred = NB_pipeline.predict(dev_data)
    dev_data_predict_time_end = time.time()
    print(f"\tNB predict time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Under and Over predictions
    underpred = np.sum(nb_dev_pred < dev_hours)
    overpred = np.sum(nb_dev_pred > dev_hours)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(dev_hours))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(dev_hours))*100}%)")

    # Test Accuracy
    dev_accuracy = accuracy_score(dev_hours_binned, nb_dev_pred)
    print(f"\tNB Accuracy: {100*dev_accuracy:.1f}%")



    ## Classifier Number 2 - Random Forest Classifier ##
    print('\nPerforming Random Forest on the reviews')

    # Create RF pipeline
    class_pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Change to desired classifier
    ])

    # Train Classifier
    RF_time_start = time.time()
    class_pipeline.fit(train_data, train_hours_binned)
    RF_time_end = time.time()
    print(f"\tRF train time: {RF_time_end - RF_time_start:.6f} seconds")

    # Predict Developer Data
    dev_data_predict_time_start = time.time()
    dev_preds = class_pipeline.predict(dev_data)
    dev_data_predict_time_end = time.time()
    print(f"\tRF predict time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")


    dev_accuracy = accuracy_score(dev_hours_binned, dev_preds)
    print(f"\tRF Accuracy: {100 * dev_accuracy:.1f}%")


if __name__ == "__main__":
    complete_classifier()
    # classify_based_on_dates()