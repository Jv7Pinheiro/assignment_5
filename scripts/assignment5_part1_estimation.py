import sys, os, warnings

import gzip
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.preprocessing import KBinsDiscretizer # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import FunctionTransformer, StandardScaler # type: ignore

import utils
warnings.filterwarnings("ignore")

NUMBER_OF_BINS = 2
REMOVE_OUTLIERS = True
LOG_TRANSFORM = True

def log_transform(labels) -> np.ndarray:
    return np.log2(labels + 1)
    
def compute_text_length(text_list: pd.Series):
    return text_list.apply(len).values.reshape(-1, 1)

def main():
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



    #############################
    ## Multinomial Naive Bayes ##
    #############################
    # We are feature engineering for regression,
    # This means that the output for NB will be used as a feature in the regression step)
    # Make prediction based on Multinomial Naive Bayes
    print('\nPerforming Multinomial Naive Bayes on the reviews')

    # Create pipeline
    NB_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            strip_accents='ascii', 
            lowercase=True, 
            analyzer='word', 
            stop_words='english')),
        ('classifier', MultinomialNB()),
    ])

    # Train data
    NB_time_start = time.time()
    NB_pipeline.fit(train_texts, train_hours_binned)
    NB_time_end = time.time()
    print(f"\tNB train time: {NB_time_end - NB_time_start:.6f} seconds")

    # Predict training data (output of this will be used to train regressor)
    train_data_predict_time_start = time.time()
    nb_train_pred = NB_pipeline.predict(train_texts)
    train_data["nb_pred"] = nb_train_pred
    train_data_predict_time_end = time.time()
    print(f"\tNB predict time (on training data): {train_data_predict_time_end - train_data_predict_time_start:.6f} seconds")

    # Predict developer data (not only to use as input to regressor, but to also measure raw performance of NB)
    dev_data_predict_time_start = time.time()
    nb_dev_pred = NB_pipeline.predict(dev_texts)
    dev_data["nb_pred"] = nb_dev_pred
    dev_data_predict_time_end = time.time()
    print(f"\tNB predict time (on developer data): {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Test accuracy of NB
    dev_accuracy = accuracy_score(dev_hours_binned, nb_dev_pred)
    print(f"\tNB Accuracy on dev set: {100*dev_accuracy:.1f}%")



    ################
    ## Estimation ##
    ################
    print("\nPerforming Linear Regression on the ['nb_pred', 'text_length', 'early_access'] features")

    if REMOVE_OUTLIERS:
        # Remove data outside the 90% percentile
        threshold = np.percentile(train_hours, 90)

        keep_mask = (train_hours <= threshold)
        train_data = train_data[keep_mask]
        train_hours = train_hours[keep_mask]

    if LOG_TRANSFORM:
        # TODO: Explain why we ravel dev but not train
        train_hours = log_transform(train_hours)
        dev_hours = log_transform(dev_hours).ravel()

    # TODO: Write a comment to explain what the hell is going on here?
    est_pipeline = Pipeline([
        ('features', ColumnTransformer(transformers=[('text_length', Pipeline([('compute_length', FunctionTransformer(compute_text_length, validate=False)), ('scaler', StandardScaler()) ]), 'text'), ('passthrough', 'passthrough', ['nb_pred', 'early_access'])])),
        ('regressor', LinearRegression())
    ])

    # Train Regressor
    REG_time_start = time.time()
    est_pipeline.fit(train_data, train_hours)
    REG_time_end = time.time()
    print(f"\tLinear Reg training time: {REG_time_end - REG_time_start:.6f} seconds")

    # Predict Dev hours using regressor
    REG_predict_time_start = time.time()
    est_dev_pred = est_pipeline.predict(dev_data)
    REG_predict_time_end = time.time()
    print(f"\tLinear Reg predict time: {REG_predict_time_end - REG_predict_time_start:.6f} seconds\n")

    if LOG_TRANSFORM:
        est_dev_pred = np.power(2, np.maximum(est_dev_pred, 0)) - 1

    # Gather and print information about regressor's estimations
    dev_mse = mean_squared_error(dev_hours, est_dev_pred) 
    dev_rmse = np.sqrt(dev_mse)
    underpred = np.sum(est_dev_pred < dev_hours)
    overpred = np.sum(est_dev_pred > dev_hours)

    print(f"\tRegressor MSE on dev dataset: {dev_rmse:.2f}")
    print(f"\tUnderpredictions: {underpred}")
    print(f"\tUnderprediction rate: {(underpred / len(dev_hours))*100}%")
    print(f"\tOverpredictions:  {overpred}")
    print(f"\tOverprediction rate rate: {(overpred / len(dev_hours))*100}%")


if __name__ == "__main__":
    main()