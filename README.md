# Amazon Sales Prediction

Utilizing PySpark to explore real-world Amazon sales and review data and predict user ratings, this project integrates AWS services (S3 and EC2) to execute a range of analytical tasks and generate aggregated outputs.

## Motivation

- Develop proficiency in PySpark for distributed data processing and analytics
- Integrate PySpark with AWS services (S3 and EC2) to build scalable big data solutions
- Gain hands-on experience by applying advanced feature engineering and predictive modeling techniques to real-world Amazon sales and review data

## Features

- **Review Statistics Aggregation:**  
  Compute average ratings, variances, and review counts for each product, enabling detailed insights into customer feedback

- **Product Category Processing:**  
  Extract and process primary category information and sales ranking details from nested data structures

- **Related Pricing Analysis:**  
  Analyze pricing data of products frequently viewed together, helping to uncover cross-sell opportunities

- **Data Imputation:**  
  Impute missing product prices using mean and median values, and handle missing product titles by assigning default values

- **Title Embedding Generation:**  
  Generate Word2Vec embeddings from product titles to capture semantic relationships and improve downstream analysis

- **Category Encoding & Dimensionality Reduction:**  
  Convert categorical product data into numerical form using one-hot encoding and reduce dimensionality via PCA

- **Regression Modeling:**  
  Build and tune decision tree regression models to predict product ratings with high accuracy

*Note: S3 paths in this project are anonymized to protect privacy.*
