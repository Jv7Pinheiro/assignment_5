import sys, warnings

import gzip
import time
import joblib
import numpy as np


from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer, StandardScaler # type: ignore

import utils
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

CLEANED_DATA_PATH = '../data/steam_reviews_clean_data.json'
NUMBER_OF_BINS = 2
REMOVE_OUTLIERS = True

def compute_text_length(X):
    return np.array([len(text) for text in X]).reshape(-1, 1)

def complete_classifier():
    # Load File
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
    print("Starting\n")
    # Append Data
    appending_time_start = time.time()
    dataset = []
    for i, l in enumerate(input_file):
        # if i > 1000:
        #     break
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
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    dev_data   = dataset[split_index:]


    # Extract Relevant Information From Data
    train_texts = train_data['text'].tolist()
    train_hours = train_data['hours'].values.reshape(-1, 1)

    dev_texts = dev_data['text'].tolist()
    dev_hours = dev_data['hours'].values.reshape(-1, 1)

    if REMOVE_OUTLIERS:
        # Remove data outside the 80% percentile
        threshold = np.percentile(train_hours, 80)

        keep_mask = (train_hours <= threshold)
        train_data = train_data[keep_mask]
        train_hours = train_hours[keep_mask]



    # Discretize the hours into quartiles (bins) of both training and dev datasets
    discretizer = KBinsDiscretizer(n_bins=NUMBER_OF_BINS, encode='ordinal', strategy='quantile')

    train_discretizer_time_start = time.time()
    train_hours_binned = discretizer.fit_transform(train_hours.reshape(-1, 1)).ravel().astype(int)
    train_discretizer_time_end = time.time()
    print(f"Train labels discretizer transform time for {NUMBER_OF_BINS} bins: {train_discretizer_time_end - train_discretizer_time_start:.6f} seconds\n")

    dev_discretizer_time_start = time.time()
    dev_hours_binned = discretizer.transform(dev_hours.reshape(-1, 1)).ravel().astype(int)
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

    best_accuracy = -1
    save_pipeline = None
    best_model_name = "None"



    ## Classifier Number 0 - Dummy classifier (random guess) %%
    print("\nPerforming Dummy Classifier (Random Guessing)")

    # Generate random predictions between 0 and (NUMBER_OF_BINS - 1)
    dummy_dev_pred = np.random.randint(0, NUMBER_OF_BINS, size=len(dev_hours))

    # Compute accuracy
    dummy_accuracy = accuracy_score(dev_hours_binned, dummy_dev_pred)
    print(f"\tDummy Classifier Accuracy: {100 * dummy_accuracy:.1f}%")



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
    underpred = np.sum(nb_dev_pred < dev_hours_binned)
    overpred = np.sum(nb_dev_pred > dev_hours_binned)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(dev_hours_binned))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(dev_hours_binned))*100}%)")

    # Test Accuracy
    dev_accuracy = accuracy_score(dev_hours_binned, nb_dev_pred)
    print(f"\tNB Accuracy: {100*dev_accuracy:.1f}%")

    if dev_accuracy > best_accuracy:
        save_pipeline =  NB_pipeline
        best_model_name = "NaiveBayes"



    ## Classifier Number 2 - Random Forest Classifier ##
    print('\nPerforming Random Forest on the reviews')

    # Create RF pipeline
    RF_pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=42))  # Change to desired classifier
    ])

    # Train Classifier
    RF_time_start = time.time()
    RF_pipeline.fit(train_data, train_hours_binned)
    RF_time_end = time.time()
    print(f"\tRF train time: {RF_time_end - RF_time_start:.6f} seconds")

    # Predict Developer Data
    dev_data_predict_time_start = time.time()
    rf_dev_preds = RF_pipeline.predict(dev_data)
    dev_data_predict_time_end = time.time()
    print(f"\tRF predict time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Under and Over predictions
    underpred = np.sum(rf_dev_preds < dev_hours_binned)
    overpred = np.sum(rf_dev_preds > dev_hours_binned)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(dev_hours_binned))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(dev_hours_binned))*100}%)")

    dev_accuracy = accuracy_score(dev_hours_binned, rf_dev_preds)
    print(f"\tRF Accuracy: {100 * dev_accuracy:.1f}%")

    if dev_accuracy > best_accuracy:
        save_pipeline =  RF_pipeline
        best_model_name = "RandomForest"



    ## Classifier Number 3 - Gradiant Boosting ##
    print('\nPerforming Gradient Boosting on the reviews')

    # Create the GB pipeline
    GB_pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, random_state=42))
    ])

    gbc_time_start = time.time()
    GB_pipeline.fit(train_data, train_hours_binned)
    gbc_time_end = time.time()
    print(f"\tGBC train time: {gbc_time_end - gbc_time_start:.6f} seconds")

    dev_data_predict_time_start = time.time()
    gbc_dev_pred = GB_pipeline.predict(dev_data)
    dev_data_predict_time_end = time.time()
    print(f"\tGBC train time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Under and Over predictions
    underpred = np.sum(gbc_dev_pred < dev_hours_binned)
    overpred = np.sum(gbc_dev_pred > dev_hours_binned)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(dev_hours_binned))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(dev_hours_binned))*100}%)")

    dev_accuracy = accuracy_score(dev_hours_binned, gbc_dev_pred)
    print(f"\tGBC Accuracy: {100 * dev_accuracy:.1f}%")

    if dev_accuracy > best_accuracy:
        save_pipeline =  GB_pipeline
        best_model_name = "GradientBoosting"

    joblib.dump(save_pipeline, f"../output/{best_model_name}_model.pkl")

def classify_based_on_dates():
    # Load File
    input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")

    print("Classifting based on hours\n\n")
    # Append Data
    appending_time_start = time.time()
    dataset = []
    for i, l in enumerate(input_file):
        # if i > 1000:
        #     break
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

    # code to extract dates
    dates = []
    for i in range(len(data)):
        dates.append(int(data.iloc[i]['date'][:4]))

    data['year'] = dates
    # Assuming your DataFrame is named df and the column containing years is 'year'
    df_before_or_during_2014 = data[data['year'] <= 2014]
    df_during_or_after_2015 = data[data['year'] >= 2015]

    df_before_or_during_2014_hours = df_before_or_during_2014['hours'].values.reshape(-1, 1)
    df_during_or_after_2015_hours = df_during_or_after_2015['hours'].values.reshape(-1, 1)

    # Discretize the hours into quartiles (bins) of both training and dev datasets
    discretizer = KBinsDiscretizer(n_bins=NUMBER_OF_BINS, encode='ordinal', strategy='quantile')

    train_discretizer_time_start = time.time()
    df_before_or_during_2014_hours_binned = discretizer.fit_transform(df_before_or_during_2014_hours.reshape(-1, 1)).ravel().astype(int)
    train_discretizer_time_end = time.time()
    print(f"Train labels discretizer transform time for {NUMBER_OF_BINS} bins: {train_discretizer_time_end - train_discretizer_time_start:.6f} seconds\n")

    dev_discretizer_time_start = time.time()
    df_during_or_after_2015_hours_binned = discretizer.transform(df_during_or_after_2015_hours.reshape(-1, 1)).ravel().astype(int)
    dev_discretizer_time_end = time.time()
    print(f"Dev labels discretizer transform time: {dev_discretizer_time_end - dev_discretizer_time_start:.6f} seconds\n")

    
    


    # Feature Processing Pipeline
    feature_transformer = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(strip_accents='ascii', lowercase=True, analyzer='word', stop_words='english'), 'text'),
        ('text_length', Pipeline([
            ('compute_length', FunctionTransformer(compute_text_length, validate=False))
        ]), 'text'),
        ('early_access', 'passthrough', ['early_access'])
    ])

    # Create MNB pipeline
    NB_pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', MultinomialNB()),
    ])

    ########################################
    ## TRAIN ON < 2014, PREDICT ON 2015 > ##
    ####################################$$$$
    print('\nTraining on < 2014, predicting on 2015 >')

    # Train Classifier
    NB_time_start = time.time()
    NB_pipeline.fit(df_before_or_during_2014, df_before_or_during_2014_hours_binned)
    NB_time_end = time.time()
    print(f"\tNB train time: {NB_time_end - NB_time_start:.6f} seconds")

    # Predict Developer Data
    dev_data_predict_time_start = time.time()
    nb_dev_pred = NB_pipeline.predict(df_during_or_after_2015)
    dev_data_predict_time_end = time.time()
    print(f"\tNB predict time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Under and Over predictions
    underpred = np.sum(nb_dev_pred < df_during_or_after_2015_hours_binned)
    overpred = np.sum(nb_dev_pred > df_during_or_after_2015_hours_binned)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(df_during_or_after_2015_hours_binned))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(df_during_or_after_2015_hours_binned))*100}%)")

    # Test Accuracy
    dev_accuracy = accuracy_score(df_during_or_after_2015_hours_binned, nb_dev_pred)
    print(f"\tNB Accuracy: {100*dev_accuracy:.1f}%")



    ########################################
    ## TRAIN ON 2015 >, PREDICT ON < 2014 ##
    ####################################$$$$
    print('\nTraining on 2015 >, predicting on < 2014')

    # Train Classifier
    NB_time_start = time.time()
    NB_pipeline.fit(df_during_or_after_2015, df_during_or_after_2015_hours_binned)
    NB_time_end = time.time()
    print(f"\tNB train time: {NB_time_end - NB_time_start:.6f} seconds")

    # Predict Developer Data
    dev_data_predict_time_start = time.time()
    nb_dev_pred = NB_pipeline.predict(df_before_or_during_2014)
    dev_data_predict_time_end = time.time()
    print(f"\tNB predict time: {dev_data_predict_time_end - dev_data_predict_time_start:.6f} seconds")

    # Under and Over predictions
    underpred = np.sum(nb_dev_pred < df_before_or_during_2014_hours_binned)
    overpred = np.sum(nb_dev_pred > df_before_or_during_2014_hours_binned)
    print(f"\tUnderpredictions: {underpred} ({(underpred / len(df_before_or_during_2014_hours_binned))*100}%)")
    print(f"\tOverpredictions:  {overpred} ({(overpred / len(df_before_or_during_2014_hours_binned))*100}%)")

    # Test Accuracy
    dev_accuracy = accuracy_score(df_before_or_during_2014_hours_binned, nb_dev_pred)
    print(f"\tNB Accuracy: {100*dev_accuracy:.1f}%")


if __name__ == "__main__":
    # complete_classifier()
    classify_based_on_dates()