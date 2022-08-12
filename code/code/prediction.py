#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import sklearn.metrics
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt

def transVolumeData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[6:12]),r[15]]).toDF(['features','label'])

def transPriceData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[9:14]),r[14]]).toDF(['features','label'])

def transPopData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[6:16]),r[16]]).toDF(['features','label'])

# file
filename='Bitcoin'

spark = SparkSession  .builder  .appName("Python Spark SQL basic example")  .config("spark.some.config.option", "some-value")  .getOrCreate()


# Load a file.
df = spark.read.format('com.databricks.spark.csv').                       options(header='true',                        inferschema='true').            load("./processed_data/"+filename+".csv",header=True);


# Select label and feature
transformed_Volume= transVolumeData(df)
transformed_Price=transPriceData(df)
transformed_Pop=transPopData(df)
# transformed_Volume.show()
# transformed_Pop.show()
# transformed_Price.show()


# Normalization
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
k_Volume = normalizer.transform(transformed_Volume)

k_Volume=k_Volume.select("normFeatures", "label")
k_Volume=k_Volume.selectExpr("normFeatures as features","label as label")
transformed_Volume=k_Volume
# transformed_Volume.show()

k_Price = normalizer.transform(transformed_Price)
k_Price=k_Price.select("normFeatures", "label")
k_Price=k_Price.selectExpr("normFeatures as features","label as label")
transformed_Price=k_Price
# transformed_Price.show()

k_Pop = normalizer.transform(transformed_Pop)
k_Pop=k_Pop.select("normFeatures", "label")
k_Pop=k_Pop.selectExpr("normFeatures as features","label as label")
transformed_Pop=k_Pop
# transformed_Pop.show()



for i in range(50):
    turn=str(i)
     # Split the data into training and test sets (30% held out for testing)
    (trainingData_Volume, testData_Volume) = transformed_Volume.randomSplit([0.7, 0.3])
    (trainingData_Price, testData_Price) = transformed_Price.randomSplit([0.7, 0.3])
    (trainingData_Pop, testData_Pop) = transformed_Pop.randomSplit([0.7, 0.3])
    # trainingData_Volume.show()
    # testData_Volume.show()
    # trainingData_Pop.show()
    # testData_Pop.show()   

    # Random Forest
    rf = RandomForestRegressor()
    pipeline = rf
    model_Volume = pipeline.fit(trainingData_Volume)
    model_Price = pipeline.fit(trainingData_Price)
    model_Pop = pipeline.fit(trainingData_Pop)

    # Prediction
    predictions_Volume = model_Volume.transform(testData_Volume)
    predictions_Price = model_Price.transform(testData_Price)
    predictions_Pop = model_Pop.transform(testData_Pop)


    # Evaluation Feature Importance
    featureI_Volume=model_Volume.featureImportances
    featureI_Price=model_Price.featureImportances
    featureI_Pop=model_Pop.featureImportances
    df_feature_importance=pd.DataFrame(columns=('A','B','C'))
    df_feature_importance_series = pd.Series({"A":featureI_Volume,"B":featureI_Price,"C":featureI_Pop},name=filename)
    df_feature_importance = df_feature_importance.append(df_feature_importance_series)
    df_feature_importance.to_csv('./results/feature_importance.csv', mode='a', header=False)


    # Evaluate RSME for Test data
    evaluator_Volume = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_Volume = evaluator_Volume.evaluate(predictions_Volume)
#     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_Volume)
    evaluator_Price = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_Price = evaluator_Price.evaluate(predictions_Price)
#     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_Price)
    evaluator_Pop = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_Pop = evaluator_Pop.evaluate(predictions_Pop)
#     print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_Pop)



    # Evaluate RMSE for Training data
    predictions_trainVolume = model_Volume.transform(trainingData_Volume)
    predictions_trainPrice = model_Price.transform(trainingData_Price)
    predictions_trainPop = model_Pop.transform(trainingData_Pop)
    evaluator_trainVolume = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_trainVolume = evaluator_trainVolume.evaluate(predictions_trainVolume)
#     print("Root Mean Squared Error (RMSE) on train data = %g" % rmse_trainVolume)
    evaluator_trainPrice = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_trainPrice = evaluator_trainPrice.evaluate(predictions_trainPrice)
#     print("Root Mean Squared Error (RMSE) on train data = %g" % rmse_trainPrice)
    evaluator_trainPop = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_trainPop = evaluator_trainPop.evaluate(predictions_trainPop)
#     print("Root Mean Squared Error (RMSE) on train data = %g" % rmse_trainPop)



    # # Evaluate R2 SCORE RSME for Test data
    y_true_Volume=predictions_Volume.select("label")
    y_pred_Volume=predictions_Volume.select("prediction")

    y_true_Price=predictions_Price.select("label")
    y_pred_Price=predictions_Price.select("prediction")

    y_true_Pop=predictions_Pop.select("label")
    y_pred_Pop=predictions_Pop.select("prediction")
    r2_score_Volume = sklearn.metrics.r2_score(y_true_Volume.select("label").collect(), y_pred_Volume.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score_Volume))
    r2_score_Price = sklearn.metrics.r2_score(y_true_Price.select("label").collect(), y_pred_Price.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score_Price))
    r2_score_Pop = sklearn.metrics.r2_score(y_true_Pop.select("label").collect(), 
                                            y_pred_Pop.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score_Pop))



    # # Evaluate R2 SCORE RSME for Train data
    y_true_trainVolume=predictions_trainVolume.select("label")
    y_pred_trainVolume=predictions_trainVolume.select("prediction")
    # y_true_trainVolume.show()
    # y_pred_trainVolume.show()
    y_true_trainPrice=predictions_trainPrice.select("label")
    y_pred_trainPrice=predictions_trainPrice.select("prediction")
    # y_true_trainPrice.show()
    # y_pred_trainPrice.show()
    y_true_trainPop=predictions_trainPop.select("label")
    y_pred_trainPop=predictions_trainPop.select("prediction")
    r2_score2_trainVolume = sklearn.metrics.r2_score(y_true_trainVolume.select("label").collect(), y_pred_trainVolume.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score2_trainVolume))
    r2_score2_trainPrice = sklearn.metrics.r2_score(y_true_trainPrice.select("label").collect(), y_pred_trainPrice.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score2_trainPrice))
    r2_score2_trainPop = sklearn.metrics.r2_score(y_true_trainPop.select("label").collect(), 
                                                  y_pred_trainPop.select("prediction").collect())
#     print('r2_score: {:4.3f}'.format(r2_score2_trainPop))



    # Output Results
    df=pd.DataFrame(columns=('A','B','C','D','E','F','G','H','I','J','K','L'))
    series = pd.Series({"A":rmse_Volume,"B":rmse_Price,"C":rmse_Pop,"D":rmse_trainVolume,
                        "E":rmse_trainPrice,"F":rmse_trainPop,"G":r2_score_Volume,"H":r2_score_Price,
                       "I": r2_score_Pop,"J":r2_score2_trainVolume , "K":r2_score2_trainPrice,  "L":r2_score2_trainPop},name=filename)
    df = df.append(series)
    df.to_csv('./results/data_results.csv', mode='a', header=False)


    # plot scatter

    # Test Volume
    List_true_Volume=y_true_Volume.select("label").collect()
    List_predict_Volume=y_pred_Volume.select("prediction").collect()
    plt.title("Test Volume")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_Volume,List_predict_Volume)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TestVolume/'+filename+'_TestVolume_'+turn+'.jpg')
#     plt.show()

    # Test Price
    List_true_Price=y_true_Price.select("label").collect()
    List_predict_Price=y_pred_Price.select("prediction").collect()
    plt.title("Test Price")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_Price,List_predict_Price)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TestPrice/'+filename+'_TestPrice_'+turn+'.jpg')
#     plt.show()

    # Test popularity
    List_true_Pop=y_true_Pop.select("label").collect()
    List_predict_Pop=y_pred_Pop.select("prediction").collect()
    plt.title("Test Popularity")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_Pop,List_predict_Pop)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TestPop/'+filename+'_TestPop_'+turn+'.jpg')
#     plt.show()


    # Test Volume
    List_true_trainVolume=y_true_trainVolume.select("label").collect()
    List_predict_trainVolume=y_pred_trainVolume.select("prediction").collect()
    plt.title("Train Volume")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_trainVolume,List_predict_trainVolume)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TrainVolume/'+filename+'_TrainVolume_'+turn+'.jpg')
#     plt.show()

    # Test Price
    List_true_trainPrice=y_true_trainPrice.select("label").collect()
    List_predict_trainPrice=y_pred_trainPrice.select("prediction").collect()
    plt.title("Train Price")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_trainPrice,List_predict_trainPrice)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TrainPrice/'+filename+'_TrainPrice_'+turn+'.jpg')
#     plt.show()

    # Test popularity
    List_true_trainPop=y_true_trainPop.select("label").collect()
    List_predict_trainPop=y_pred_trainPop.select("prediction").collect()
    plt.title("Train Popularity")
    plt.xlabel("True value")
    plt.ylabel("Predictions")
    plt.scatter(List_true_trainPop,List_predict_trainPop)
#     plt.savefig('BigDataCW2/BigdataFigure/'+filename+'/TrainPop/'+filename+'_TrainPop_'+turn+'.jpg')
#     plt.show()





# In[ ]:


print("done")

