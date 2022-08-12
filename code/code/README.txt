-> data_preprocessing.py
1. Test  another dataset
The first line of the main code defines a string: filename. If you want to test another dataset, you can just change the string to the name of other dataset. For example, now the filename is 'Bitcoin', if you want to run the dataset 'Tether'. You can just change the 'Bitcoin' to ‘Tether'.

-> prediction.py
1. Test  another dataset
The first line of the main code(just after the fuction definitions) defines a string: filename. If you want to test another dataset, you can just change the string to the name of other dataset. For example, now the filename is 'Bitcoin', if you want to run the dataset 'Tether'. You can just change the 'Bitcoin' to ‘Tether'.


The codes in prediction.py refers to:
1. Random forest regression: https://runawayhorse001.github.io/LearningApacheSpark/regression.html#random-forest-regression
2. Normalization: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.feature.Normalizer.html
3. RDD map: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.map.html
4. Feature importance: http://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassificationModel.html?highlight=featureimportance#pyspark.ml.classification.DecisionTreeClassificationModel.featureImportances

