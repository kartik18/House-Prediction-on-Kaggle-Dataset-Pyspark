from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import numpy as np
import pyspark.sql.functions

sc = SparkContext("local", "Spark Demo")
# print(sc.textFile("D:/words.txt").first())
sqlContext = SQLContext(sc)

housing_df = sqlContext.read.format("csv")\
    .options(delimiter=',', header=True, inferSchema=True)\
    .load("D:/house_data.csv")

# housing_df.show(5)

housing_df.cache()

# housing_df.printSchema()

# print(housing_df.count())
housing_df.describe("price", "sqft_living").show()
# housing_df.select("bedrooms","price").show(5)
# housing_df.select([max('price'),min('price'])
# df = sqlContext.range(0, 10).withColumn('rand1', rand(seed=10)).withColumn('rand2', rand(seed=27))
# print(df.stat.cov('rand1', 'rand2'))

bed_bath_df = housing_df.select('waterfront', 'view')
# bed_bath_df.show(5)

# bed_bath_df.stat.crosstab('view', 'waterfront').show()

price_condition_df = housing_df.groupBy('condition').avg('price')

price_condition_df.show()

housing_price_pd = housing_df.select('price').toPandas()
# print("Lenght of DF", len(housing_price_pd.index))
count_housing_price_pd = len(housing_price_pd.index)


# To find the correlation between two entities
print("The correlation b/w price & square foot living ", housing_df.stat.corr('price', 'sqft_living'))

column_labels = ['price','sqft_living', 'sqft_lot', 'bedrooms','bathrooms',
         'floors', 'sqft_above', 'sqft_basement','yr_built','yr_renovated',
        'sqft_living15', 'sqft_lot15']

column_corr = Statistics.corr(housing_df.rdd.map(lambda x:
                                                 np.array([x['price'],
                                                           x['sqft_living'],
                                                           x['sqft_lot'],
                                                           x['bedrooms'],
                                                           x['bathrooms'],
                                                           x['floors'],
                                                           x['sqft_above'],
                                                           x['sqft_basement'],
                                                           x['yr_built'],
                                                           x['yr_renovated'],
                                                           x['sqft_living15'],
                                                           x['sqft_lot15']
                                                           ])), method='pearson')
#print(column_corr)

price_by_zipcode_df = housing_df.groupBy('zipcode').avg('price')

# price_by_zipcode_df.show()

housing_df_clean = housing_df.na.drop(how='any')

print("Bool that is contains no null values", housing_df_clean.count() == housing_df.count())

## Correcting the price column
housing_df = housing_df.withColumn("log_price", pyspark.sql.functions.log('price'))

## Correcting the Sqft_lot
housing_df = housing_df.withColumn("log_sqft_lot", pyspark.sql.functions.log('sqft_lot'))

print("Correlation b/w price and sqft_lot", housing_df.stat.corr('price', 'sqft_lot'))
print("Correlation b/w log_price and log_sqft_lot", housing_df.stat.corr('log_sqft_lot', 'log_price'))

## Calculating age of the house. Taking 2018 as the base
from pyspark.sql.functions import lit, col
housing_df = housing_df.withColumn("age",lit(2018) - col('yr_built'))
#housing_df.show(5)

## When was the house renovated
housing_df = housing_df.withColumn("rennovate_age", lit(2018) - col('yr_renovated'))

housing_df.show(5)

housing_original_df = housing_df

continuous_features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors',
                    'log_sqft_lot', 'age', 'sqft_above',
                    'sqft_living15', 'sqft_lot15', 'rennovate_age']

categorical_features = ['zipcode', 'waterfront', 'grade', 'condition', 'view']

# Function to create categorical features
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, PolynomialExpansion, VectorIndexer

def create_catgeory_vars(dataset, field_name):
    idx_col = field_name + "Index"
    col_vec = field_name + "Vec"
    month_stringindexer = StringIndexer(inputCol=field_name, outputCol=idx_col)
    month_model = month_stringindexer.fit(dataset)
    month_indexed = month_model.transform(dataset)
    month_encoder = OneHotEncoder(dropLast=True, inputCol=idx_col, outputCol=col_vec)

    return month_encoder.transform(month_indexed)


for col in categorical_features:
    housing_df = create_catgeory_vars(housing_df, col)

print("--")
print(housing_df.printSchema())

housing_df.show(4)

featureCols = continuous_features + ['zipcodeVec', 'waterfrontVec', 'gradeVec', 'conditionVec', 'viewVec']

# print(featureCols)

assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# print(assembler.outputCol)
print("===|||===")
housing_train_df = assembler.transform(housing_df)

housing_train_df.show(5)
from pyspark.sql.functions import round
housing_train_df = housing_train_df.withColumn("label", round('log_price', 4))

print("-----%%-----")

housing_train_df.show(5)

seed = 42

train_df, test_df = housing_train_df.randomSplit( [0.7, 0.3], seed=seed)

from pyspark.ml.regression import LinearRegression

linreg = LinearRegression(maxIter=500, regParam=0.0)

lm = linreg.fit(train_df)

print("Intercept ", lm.intercept)
print("Coefficients ", lm.coefficients)

y_pred = lm.transform(test_df)

y_pred.select('features', 'label', 'prediction').show(5)

from pyspark.sql.functions import exp

y_pred = y_pred.withColumn("y_pred", exp('prediction'))

y_pred.show(5)

from pyspark.ml.evaluation import RegressionEvaluator

rmse_evaluator = RegressionEvaluator(labelCol="price",
                                     predictionCol="y_pred",
                                     metricName="rmse")

lm_rmse = rmse_evaluator.evaluate(y_pred)

print("Root mean square ", lm_rmse)

rsquare_evaluator = RegressionEvaluator(labelCol="price",
                                        predictionCol="y_pred",
                                        metricName="r2")

lm_rsquare = rsquare_evaluator.evaluate( y_pred)

print("R square ", lm_rsquare)