import os
import json
import boto3
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml as M
from pyspark.sql import SparkSession

def compute_review_statistics(results_storage, review_data, product_data):
    # Aggregate review statistics (mean and count) per product (asin)
    asin_stats = review_data.groupBy('asin').agg(
        F.avg('overall').alias('meanRating'),
        F.count('overall').alias('countRating')
    )
    # Join the aggregated review stats with the product data and cache for performance
    joined_df = product_data.select('asin').join(asin_stats, on='asin', how='left')
    joined_df.cache()

    res = {
        'count_total': joined_df.count(),
        'mean_meanRating': joined_df.select(F.mean(F.col('meanRating'))).collect()[0][0],
        'variance_meanRating': joined_df.select(F.variance(F.col('meanRating'))).collect()[0][0],
        'numNulls_meanRating': joined_df.filter(F.col('meanRating').isNull()).count(),
        'mean_countRating': joined_df.select(F.mean(F.col('countRating'))).collect()[0][0],
        'variance_countRating': joined_df.select(F.variance(F.col('countRating'))).collect()[0][0],
        'numNulls_countRating': joined_df.filter(F.col('countRating').isNull()).count()
    }
    joined_df.unpersist()
    results_storage['compute_review_statistics'] = res
    return res

def process_product_categories(results_storage, product_data):
    # Extract the primary category from the nested categories list
    category_col = F.when(
        (F.col('categories').isNull()) | (F.size(F.col('categories')) == 0) | (F.col('categories')[0][0] == ''),
        None
    ).otherwise(F.col('categories')[0][0]).alias('category')

    # Extract the best sales category and corresponding rank from the salesRank map
    bestSalesCategory_col = F.when(
        (F.col('salesRank').isNull()) | (F.size(F.map_keys(F.col('salesRank'))) == 0),
        None
    ).otherwise(F.map_keys(F.col('salesRank'))[0]).alias('bestSalesCategory')

    bestSalesRank_col = F.when(
        (F.col('salesRank').isNull()) | (F.size(F.map_values(F.col('salesRank'))) == 0),
        None
    ).otherwise(F.map_values(F.col('salesRank'))[0]).alias('bestSalesRank')

    df = product_data.select('asin', category_col, bestSalesCategory_col, bestSalesRank_col)
    res = {
        'count_total': df.count(),
        'mean_bestSalesRank': df.select(F.mean(F.col('bestSalesRank'))).collect()[0][0],
        'variance_bestSalesRank': df.select(F.variance(F.col('bestSalesRank'))).collect()[0][0],
        'numNulls_category': df.filter(F.col('category').isNull()).count(),
        'countDistinct_category': df.select(F.col('category')).distinct().count() - 1,
        'numNulls_bestSalesCategory': df.filter(F.col('bestSalesCategory').isNull()).count(),
        'countDistinct_bestSalesCategory': df.select(F.col('bestSalesCategory')).distinct().count() - 1
    }
    results_storage['process_product_categories'] = res
    return res

def compute_related_pricing(results_storage, product_data):
    # Explode the list of "also_viewed" ASINs to join with product pricing information
    exploded_df = product_data.select(
        F.col('asin'),
        F.explode_outer(F.col('related.also_viewed')).alias('also_viewed_asin')
    )
    # Self-join to retrieve the price of each "also_viewed" product
    exploded_df = exploded_df.join(
        product_data.select(F.col('asin').alias('also_viewed_asin'),
                              F.col('price').alias('also_viewed_price')),
        on="also_viewed_asin",
        how="left"
    )
    # Aggregate by computing the mean price and count of "also_viewed" items per product
    agg_df = exploded_df.groupby('asin').agg(
        F.mean(F.col('also_viewed_price')).alias('meanPriceAlsoViewed'),
        F.count(F.col('also_viewed_asin')).alias('countAlsoViewed')
    )
    # Replace zero counts with null to indicate absence of data
    agg_df = agg_df.withColumn('countAlsoViewed',
                               F.when(F.col('countAlsoViewed') == 0, None).otherwise(F.col('countAlsoViewed')))
    agg_df.cache()

    res = {
        'count_total': agg_df.count(),
        'mean_meanPriceAlsoViewed': agg_df.select(F.mean(F.col('meanPriceAlsoViewed'))).collect()[0][0],
        'variance_meanPriceAlsoViewed': agg_df.select(F.variance(F.col('meanPriceAlsoViewed'))).collect()[0][0],
        'numNulls_meanPriceAlsoViewed': agg_df.filter(F.col('meanPriceAlsoViewed').isNull()).count(),
        'mean_countAlsoViewed': agg_df.select(F.mean(F.col('countAlsoViewed'))).collect()[0][0],
        'variance_countAlsoViewed': agg_df.select(F.variance(F.col('countAlsoViewed'))).collect()[0][0],
        'numNulls_countAlsoViewed': agg_df.filter(F.col('countAlsoViewed').isNull()).count()
    }
    agg_df.unpersist()
    results_storage['compute_related_pricing'] = res
    return res

def impute_product_fields(results_storage, product_data):
    # Ensure the price column is in float format and compute imputation statistics
    product_data = product_data.withColumn('price', F.col('price').cast('float'))
    price_stats = product_data.select(
        F.mean(F.col('price')).alias('mean_price'),
        F.expr('percentile_approx(price, 0.5)').alias('median_price')
    ).collect()[0]
    mean_price = price_stats['mean_price']
    median_price = price_stats['median_price']

    # Impute missing prices with the computed mean and median; assign 'unknown' for missing titles
    product_data = product_data.withColumn(
        'meanImputedPrice', F.when(F.col('price').isNull(), mean_price).otherwise(F.col('price'))
    ).withColumn(
        'medianImputedPrice', F.when(F.col('price').isNull(), median_price).otherwise(F.col('price'))
    ).withColumn(
        'unknownImputedTitle', F.when((F.col('title').isNull()) | (F.col('title') == ''), 'unknown').otherwise(F.col('title'))
    )
    res = {
        'count_total': product_data.count(),
        'mean_meanImputedPrice': product_data.select(F.mean(F.col('meanImputedPrice'))).collect()[0][0],
        'variance_meanImputedPrice': product_data.select(F.variance(F.col('meanImputedPrice'))).collect()[0][0],
        'numNulls_meanImputedPrice': product_data.filter(F.col('meanImputedPrice').isNull()).count(),
        'mean_medianImputedPrice': product_data.select(F.mean(F.col('medianImputedPrice'))).collect()[0][0],
        'variance_medianImputedPrice': product_data.select(F.variance(F.col('medianImputedPrice'))).collect()[0][0],
        'numNulls_medianImputedPrice': product_data.filter(F.col('medianImputedPrice').isNull()).count(),
        'numUnknowns_unknownImputedTitle': product_data.filter(F.col('unknownImputedTitle') == 'unknown').count()
    }
    results_storage['impute_product_fields'] = res
    return res

def generate_title_embeddings(results_storage, product_processed_data, word_0, word_1, word_2):
    # Convert title text to lowercase and split into an array of words
    product_processed_data = product_processed_data.withColumn(
        'titleArray', F.split(F.lower(F.col('title')), ' ')
    )
    # Train a Word2Vec model to generate vector embeddings from the title words
    word2vec = M.feature.Word2Vec(
        vectorSize=16, minCount=100, inputCol='titleArray', outputCol='titleVector',
        seed=102, numPartitions=4
    )
    word2vec_model = word2vec.fit(product_processed_data)
    product_processed_data_output = word2vec_model.transform(product_processed_data)
    res = {
        'count_total': product_processed_data_output.count(),
        'size_vocabulary': word2vec_model.getVectors().count(),
        'word_0_synonyms': word2vec_model.findSynonymsArray(word_0, 10),
        'word_1_synonyms': word2vec_model.findSynonymsArray(word_1, 10),
        'word_2_synonyms': word2vec_model.findSynonymsArray(word_2, 10)
    }
    results_storage['generate_title_embeddings'] = res
    return res

def encode_and_reduce_categories(results_storage, product_processed_data):
    # Convert category strings into numerical indices
    string_indexer = M.feature.StringIndexer(inputCol='category', outputCol='categoryIndex')
    product_processed_data = string_indexer.fit(product_processed_data).transform(product_processed_data)
    # One-hot encode the numerical indices
    one_hot_encoder = M.feature.OneHotEncoder(inputCol='categoryIndex', outputCol='categoryOneHot', dropLast=False)
    product_processed_data = one_hot_encoder.fit(product_processed_data).transform(product_processed_data)
    # Reduce dimensionality using PCA on the one-hot encoded features
    pca = M.feature.PCA(k=15, inputCol='categoryOneHot', outputCol='categoryPCA')
    product_processed_data = pca.fit(product_processed_data).transform(product_processed_data)
    product_processed_data = product_processed_data.withColumn('categoryOneHot', F.col('categoryOneHot'))
    # Compute mean vectors of the encoded features for further analysis
    summarizer = M.stat.Summarizer.metrics('mean')
    meanVector_categoryOneHot = product_processed_data.select(
        summarizer.summary(F.col('categoryOneHot'))
    ).collect()[0][0][0].toArray().tolist()
    meanVector_categoryPCA = product_processed_data.select(
        summarizer.summary(F.col('categoryPCA'))
    ).collect()[0][0][0].toArray().tolist()

    res = {
        'count_total': product_processed_data.count(),
        'meanVector_categoryOneHot': meanVector_categoryOneHot,
        'meanVector_categoryPCA': meanVector_categoryPCA
    }
    results_storage['encode_and_reduce_categories'] = res
    return res

def decision_tree_regression(results_storage, train_data, test_data):
    # Train a Decision Tree regression model with a fixed maximum depth
    dt = M.regression.DecisionTreeRegressor(featuresCol='features', labelCol='overall', maxDepth=5)
    dt_model = dt.fit(train_data)
    pred_df = dt_model.transform(test_data)
    # Calculate squared error and then RMSE as the evaluation metric
    pred_df = pred_df.withColumn('sq_err', (F.col('prediction') - F.col('overall')) ** 2)
    test_mse = pred_df.select(F.mean(F.col('sq_err'))).collect()[0][0]
    test_rmse = test_mse ** 0.5
    res = {'test_rmse': test_rmse}
    results_storage['decision_tree_regression'] = res
    return res

def validate_decision_tree(train_data, valid_data, max_depth):
    # Helper function to train and evaluate a Decision Tree model for a given depth
    dt = M.regression.DecisionTreeRegressor(featuresCol='features', labelCol='overall', maxDepth=max_depth)
    dt_model = dt.fit(train_data)
    pred_df = dt_model.transform(valid_data)
    pred_df = pred_df.withColumn('sq_err', (F.col('prediction') - F.col('overall')) ** 2)
    valid_mse = pred_df.select(F.mean(F.col('sq_err'))).collect()[0][0]
    valid_rmse = valid_mse ** 0.5
    return valid_rmse, dt_model

def dt_hyperparameter_tuning(results_storage, train_data, test_data):
    # Split the training data into training and validation sets for hyperparameter tuning
    train_data, valid_data = train_data.randomSplit([0.75, 0.25])
    dt_results = []
    for dt_depth in [5, 7, 9, 12]:
        valid_rmse, dt_model = validate_decision_tree(train_data, valid_data, dt_depth)
        dt_results.append((valid_rmse, dt_model, dt_depth))
    # Select the best model based on the lowest validation RMSE
    best_dt_model = min(dt_results, key=lambda x: x[0])
    pred_df = best_dt_model[1].transform(test_data)
    pred_df = pred_df.withColumn('sq_err', (F.col('prediction') - F.col('overall')) ** 2)
    test_mse = pred_df.select(F.mean(F.col('sq_err'))).collect()[0][0]
    test_rmse = test_mse ** 0.5
    res = {'test_rmse': test_rmse}
    for valid_rmse, _, dt_depth in dt_results:
        res[f'valid_rmse_depth_{dt_depth}'] = valid_rmse
    results_storage['dt_hyperparameter_tuning'] = res
    return res

def main():
    spark = SparkSession.builder.appName("AmazonSalesPrediction").getOrCreate()

    # S3 paths
    bucket_name = "amazon-sales-insights"
    reviews_path = f"s3://{bucket_name}/input/reviews.json"
    products_path = f"s3://{bucket_name}/input/products.json"
    products_processed_path = f"s3://{bucket_name}/input/products_processed.json"

    # Load data from S3 using Spark
    review_data = spark.read.json(reviews_path)
    product_data = spark.read.json(products_path)
    product_processed_data = spark.read.json(products_processed_path)
    train_data, test_data = product_processed_data.randomSplit([0.8, 0.2])

    # Create a dictionary to store results from all tasks
    results_storage = {}
    compute_review_statistics(results_storage, review_data, product_data)
    process_product_categories(results_storage, product_data)
    compute_related_pricing(results_storage, product_data)
    impute_product_fields(results_storage, product_data)
    generate_title_embeddings(results_storage, product_processed_data, "word_0", "word_1", "word_2")
    encode_and_reduce_categories(results_storage, product_processed_data)
    decision_tree_regression(results_storage, train_data, test_data)
    dt_hyperparameter_tuning(results_storage, train_data, test_data)

    # Save results to local file
    output_results = "results.json"
    with open(output_results, 'w') as f:
        json.dump(results_storage, f, indent=4)

    # Upload results back to S3
    s3 = boto3.client('s3') # Local IAM credentials are used
    s3.upload_file(output_results, bucket_name, "output/results.json")

if __name__ == '__main__':
    main()