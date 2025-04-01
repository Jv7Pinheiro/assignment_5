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
    # DEBUGGING Print sizes of train_data and dev_data in bytes
    # print(f"Train data size: {sum(len(pickle.dumps(d)) for d in train_data)} bytes")
    # print(f"Dev data size: {sum(len(pickle.dumps(d)) for d in dev_data)} bytes\n")

    # TODO: This should be a function inside utils that takes in data and a string that
    # corresponds to a json feature and it returns a vector of just the features of data

    nb_train_texts = train_data['text'].tolist()
    nb_train_hours = train_data['hours'].values.reshape(-1, 1)

    train_discretizer_time_start = time.time()
    discretizer = KBinsDiscretizer(n_bins=NUMBER_OF_BINS, encode='ordinal', strategy='quantile')
    nb_train_hours_binned = discretizer.fit_transform(nb_train_hours).ravel().astype(int)
    train_discretizer_time_end = time.time()
    print(f"Train labels discretizer transform time for {NUMBER_OF_BINS} bins: {train_discretizer_time_end - train_discretizer_time_start:.6f} seconds\n")
    
    # nb_train_hours_binned = np.digitize(train_hours, [2, 4, 10, 50, 100, 500, 1000], right=False)

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
    NB_pipeline.fit(nb_train_texts, nb_train_hours_binned)
    NB_time_end = time.time()
    print(f"NB training time: {NB_time_end - NB_time_start:.6f} seconds\n")

    nb_dev_texts = dev_data['text'].tolist()
    nb_dev_hours = dev_data['hours'].values.reshape(-1, 1)
    
    dev_discretizer_time_start = time.time()
    nb_dev_hours_binned = discretizer.transform(nb_dev_hours).ravel().astype(int)
    dev_discretizer_time_end = time.time()
    print(f"Dev labels discretizer transform time: {dev_discretizer_time_end - dev_discretizer_time_start:.6f} seconds\n")
    # y_dev_hours_binned = np.digitize(dev_hours, [2, 4, 10, 50, 100, 500, 1000], right=False)

    train_data_predict_time_start = time.time()
    nb_train_pred = NB_pipeline.predict(nb_train_texts)
    train_data["nb_pred"] = nb_train_pred
    train_data_predict_time_end = time.time()
    print(f"Train data NB predict time (this will be plugged in regressor training): "+
          f"{train_data_predict_time_end - train_data_predict_time_start:.6f} seconds\n")

    dev_data_predict_time_start = time.time()
    nb_dev_pred = NB_pipeline.predict(nb_dev_texts)
    dev_data["nb_pred"] = nb_dev_pred
    dev_data_predict_time_end = time.time()
    print(f"Dev data NB predict time (this will be plugged in regressor predict): "+
          f"{dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds\n")

    dev_accuracy = accuracy_score(nb_dev_hours_binned, nb_dev_pred)
    print(f"Accuracy on dev set: {100*dev_accuracy:.1f}%")

    #############################
    ## Estimation ##
    #############################

    train_data["text_length"] = train_data["text"].astype(str).apply(len)
    dev_data["text_length"] = dev_data["text"].astype(str).apply(len)

    est_X_train = train_data[["nb_pred", "text_length", "early_access"]]
    est_y_train  = train_data['hours'].values.reshape(-1, 1)
    est_y_dev = dev_data['hours'].values.reshape(-1, 1)

    if REMOVE_OUTLIERS:
        remove_outliers_time_start = time.time()
        threshold = np.percentile(est_y_train, 90)
        keep_mask = (est_y_train <= threshold)
        est_X_train = est_X_train[keep_mask]
        est_y_train = est_y_train[keep_mask]
        remove_outliers_time_end = time.time()
        print(f"Outlier removal time: {remove_outliers_time_end - remove_outliers_time_start:.6f} seconds\n")

    if LOG_TRANSFORM:
        log_transform_time_start = time.time()
        est_y_train = log_transform(est_y_train)
        log_transform_time_end = time.time()
        print(f"Log transform time: {log_transform_time_end - log_transform_time_start:.6f} seconds\n")

    est_pipeline = Pipeline([
        ('regressor', LinearRegression()),
    ])

    print("\nPerforming Linear Regression on the ['nb_pred', 'text_length', 'early_access'] features")
    REG_time_start = time.time()
    est_pipeline.fit(est_X_train, est_y_train)
    REG_time_end = time.time()
    print(f"Linear Reg training time: {REG_time_end - REG_time_start:.6f} seconds\n")

    REG_predict_time_start = time.time()
    est_dev_pred = est_pipeline.predict(dev_data[["nb_pred", "text_length", "early_access"]])
    REG_predict_time_end = time.time()
    print(f"Linear Reg predict time: {REG_predict_time_end - REG_predict_time_start:.6f} seconds\n")

    if LOG_TRANSFORM:
        log_transform_pred_time_start = time.time()
        est_dev_pred = np.power(2, np.maximum(est_dev_pred, 0)) - 1
        est_y_dev = log_transform(est_y_dev).ravel()
        log_transform_pred_time_end = time.time()
        print(f"Log transform time on prediction: {log_transform_pred_time_end - log_transform_pred_time_start:.6f} seconds\n")

    dev_mse = mean_squared_error(est_y_dev, est_dev_pred) 
    dev_rmse = np.sqrt(dev_mse)

    underpred = np.sum(est_dev_pred < est_y_dev)
    overpred = np.sum(est_dev_pred > est_y_dev)

    print(f"RMSE on dev set: {dev_rmse:.2f}\n")
    print(f"Underpredictions: {underpred}\n")
    print(f"Overpredictions:  {overpred}\n\n")


if __name__ == "__main__":
    main()