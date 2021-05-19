from __future__ import print_function

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import concat_ws
from pyspark.sql.types import StructField,IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import*

import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession\
        .builder\
        .appName("KMeansExample")\
        .getOrCreate()
​
df = spark.read.load(sys.argv[1],format="csv", sep=",", inferSchema="true", header="true")
​
#LOAD DATA
dataset = spark.read.format("libsvm").load("/mapreduce-test/pro_1/shot_logs.csv")
df = df.drop(columns=['GAME_ID','SHOT_RESULT','MATCHUP','LOCATION','W','FINAL_MARGIN','SHOT_NUMBER','PERIOD','GAME_CLOCK','DRIBBLES','CLOSEST_DEFENDER','CLOSEST_DEFENDER_PLAYER_ID','FGM','PTS','player_id','PTS_TYPE','TOUCH_TIME'])
df = df[df.SHOT_RESULT != 'missed']
df = df.groupby('player_name')
df = df.mean()

#FILTER DATA TO GROUP BY PLAYER​
training_set = x = df[['SHOT_CLOCK','SHOT_DIST','CLOSE_DEF_DIST']]

#EXPLORATION OF KVALUES

#Convert dataset into VectorRow data cells
data_of_interest = dataset.withColumn('CLOSE_DEF_DIST', data_cleaned['CLOSE_DEF_DIST'].cast(DoubleType())).withColumn('SHOT_DIST', data_cleaned['SHOT_DIST, '].cast(DoubleType())).withColumn('SHOT_CLOCK', data_cleaned['SHOT_CLOCK'].cast(DoubleType()))
feature_vector = VectorAssembler(inputCols=['CLOSE_DEF_DIST', 'SHOT_DIST', 'SHOT_CLOCK'], outputCol="features")
transform_data = feature_vector.transform(data_of_interest)
player_names = transform_data.select("player_name").distinct().collect()
list_items = list()
evaluator = ClusteringEvaluator()

#Getting Silhouette with squared euclidean distance for k value ranging from 2 to 8
TotalSED = []
for player in player_name:
    features = transform_data.where(transform_data["player_name"] == player[0]).select("features")
    for k in range(2,8):
        kmeans = KMeans(featuresCol = 'features', k=k)
        model = kmeans.fit(features)
        predictions = model.transform(features)
        silhouette = evaluator.evaluate(predictions)
        print("With K={}".format(k))
        print("Silhouette with squared euclidean distance = " + str(silhouette))
        TotalSED.append(silhouette)
    break

#plotting kvalues and Total_SED 
plt.plt(range(2,9), TSED); plt.xlabel("No_of_Clusters"); plt.ylabel("Total_SED"); plt.xticks(k)


#ESTABLISH MODEL WITH KMEANS
kmeans = KMeans().setK(4).setSeed(1)
model = kmeans.fit(training_set)
​
# Make predictions
predictions = model.transform(training_set)
​
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
​
# Shows the result.
centers = model.clusterCenters()

def e_dist(stat_matrix,cent_list):
	mat_dist = []
	for i in range(0,len(centers)):
		dist_sqrd = 0
		for j in range(0,len(centers[i])):
			dist_sqrd += (float(stat_matrix[j]) - float(cent_list[i][j]))**2
​
		dist = math.sqrt(dist_sqrd)
		mat_dist.append(dist)

spark.stop()