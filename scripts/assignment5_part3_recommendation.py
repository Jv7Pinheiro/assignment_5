import sys, os, warnings

import gzip
import time
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession # type: ignore
from pyspark.ml.feature import StringIndexer, StringIndexerModel # type: ignore
from pyspark.ml.recommendation import ALSModel, ALS # type: ignore
from pyspark.ml.evaluation import RegressionEvaluator # type: ignore
from pyspark.sql.functions import col, log, pow, greatest, lit, isnan # type: ignore
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator #type: ignore

import utils
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

LOG_TRANSFORM = True
REMOVE_OUTLIERS = True
CLEANED_DATA_PATH = '../data/steam_reviews_clean_data.json'

OUTPUT_PATH = "../output/part3"
USER_INDEXER_PATH = f"{OUTPUT_PATH}/user_indexer"
PRODUCT_INDEXER_PATH = f"{OUTPUT_PATH}/product_indexer"
ALS_MODEL_PATH = f"{OUTPUT_PATH}/als_model"

def main():
    
    if not os.path.exists(CLEANED_DATA_PATH):
        # Load File
        input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")

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
        
        cleaned_df = pd.DataFrame(data)
        cleaned_df.to_json(CLEANED_DATA_PATH, orient="records", lines=True)
        print(f"Cleaned data saved to {CLEANED_DATA_PATH}\n")

    spark = (
        SparkSession.builder
            .master("local")
            .appName("Hours Recommendation")
            .config("spark.driver.memory", "128g")
            .config("spark.executor.memory", "16g")
            .getOrCreate()
    )

    spark_df = spark.read.json(CLEANED_DATA_PATH)
    print("Cleaned data loaded into Spark DataFrame.\n")

    subset_df = spark_df.sample(fraction=0.05, seed=42)

    train_spark_df, dev_spark_df = subset_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train count: {train_spark_df.count()}, Dev count: {dev_spark_df.count()}\n")

    user_stringIndexer = StringIndexer(inputCol="username", outputCol="user_index", handleInvalid="keep")
    product_stringIndexer = StringIndexer(inputCol="product_id", outputCol="product_index", handleInvalid="keep")

    ####################
    ## Recommendation ##
    ####################

    user_model = user_stringIndexer.fit(train_spark_df)
    product_model = product_stringIndexer.fit(train_spark_df)

    train_spark_df = user_model.transform(train_spark_df)
    train_spark_df = product_model.transform(train_spark_df)

    dev_spark_df = user_model.transform(dev_spark_df)
    dev_spark_df = product_model.transform(dev_spark_df)
    
    # Remove outliers from training data
    if REMOVE_OUTLIERS:
        threshold = train_spark_df.approxQuantile("hours", [0.9], 0.01)[0]
        print(f"90th percentile threshold: {threshold:.2f} hours")
        
        train_spark_df = train_spark_df.filter(col("hours") <= threshold)
        print(f"Training count after outlier removal: {train_spark_df.count()}")

    if LOG_TRANSFORM:
        train_spark_df = train_spark_df.withColumn("hours", log(2.0, col("hours") + 1))

    # Create the ALS model
    als = ALS(
        maxIter=5,
        regParam=0.1,
        rank=50,
        seed=0,
        userCol="user_index",
        itemCol="product_index",
        ratingCol="hours",
        coldStartStrategy="drop"
    )

    # Recommender training
    REC_train_time_start = time.time()
    paramGrid = (
        ParamGridBuilder()
        .addGrid(als.maxIter, [5, 10, 15])
        .build()
    )
    evaluator = RegressionEvaluator(labelCol="hours", predictionCol="prediction", metricName="rmse")
    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cvModel = cv.fit(train_spark_df)
    als_model = cvModel.bestModel
    REC_train_time_end = time.time()
    print(f"\tRecommender training time: {REC_train_time_end - REC_train_time_start:.6f} seconds")

    # Predict Dev hours using regressor
    REC_predict_time_start = time.time()
    predictions = als_model.transform(dev_spark_df)
    REC_predict_time_end = time.time()
    print(f"\tRecommender predict time: {REC_predict_time_end - REC_predict_time_start:.6f} seconds\n")

    if LOG_TRANSFORM:
        predictions = predictions.withColumn("prediction", pow(2.0, greatest(col("prediction"), lit(0))) - 1)

    predictions = predictions.filter(
        col("prediction").isNotNull() & ~isnan("prediction") &
        col("hours").isNotNull() & ~isnan("hours")
    )

    # Gather and print information about regressor's estimations
    evaluator = RegressionEvaluator(
        labelCol="hours",
        predictionCol="prediction",
        metricName="rmse"
    )

    total = predictions.count()
    dev_rmse = evaluator.evaluate(predictions)
    underpred = predictions.filter(predictions.prediction < predictions.hours).count()
    overpred = predictions.filter(predictions.prediction > predictions.hours).count()

    print(f"\RMSE on dev dataset: {dev_rmse:.2f}")
    print(f"\tUnderpredictions: {underpred}")
    print(f"\tUnderprediction rate: {(underpred / total)*100}%")
    print(f"\tOverpredictions:  {overpred}")
    print(f"\tOverprediction rate rate: {(overpred / total)*100}%")

    user_model.write().overwrite().save(USER_INDEXER_PATH)
    product_model.write().overwrite().save(PRODUCT_INDEXER_PATH)
    als_model.write().overwrite().save(ALS_MODEL_PATH)
    print(f"\tPipeline saved.\n")

def predict_real_data(new_data_path: str, output_path: str):
    spark = SparkSession.builder.appName("Real Data Prediction").getOrCreate()

    new_df = spark.read.json(new_data_path)
    new_df = new_df.sample(fraction=0.05, seed=42)

    user_model = StringIndexerModel.load(USER_INDEXER_PATH)
    product_model = StringIndexerModel.load(PRODUCT_INDEXER_PATH)
    als_model = ALSModel.load(ALS_MODEL_PATH)

    indexed_df = user_model.transform(new_df)
    indexed_df = product_model.transform(indexed_df)

    predictions = als_model.transform(indexed_df)

    if LOG_TRANSFORM:
        predictions = predictions.withColumn("prediction", pow(2.0, greatest(col("prediction"), lit(0))) - 1)

    predictions.select("username", "product_id", "prediction").write.json(output_path, mode="overwrite")
    print(f"\nPredictions written to: {output_path}")


if __name__ == "__main__":
    main()
    # predict_real_data(
    #     new_data_path="../data/steam_reviews_clean_data.json",
    #     output_path="../output/part3/predictions.json"
    # )
