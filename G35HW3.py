from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
import random
import sys
import os
import numpy as np
import time
import math

assert len(sys.argv) == 7, "Usage: python G35HW3.py <file_name> <kstart> <h> <iter> <M> <L>"

data_path = sys.argv[1]
# assert os.path.isfile(data_path), "File or folder not found"
data_path = str(data_path)

kstart = sys.argv[2]
assert kstart.isdigit()
kstart = int(kstart)

h = sys.argv[3]
assert h.isdigit()
h = int(h)

iter = sys.argv[4]
assert iter.isdigit()
iter = int(iter)

m = sys.argv[5]
assert m.isdigit()
m = int(m)

L = sys.argv[6]
assert L.isdigit()
L = int(L)


def strToTuple(line):
    ch = line.strip().split(" ")
    point = tuple(float(ch[i]) for i in range(len(ch)))
    return point


def math_dist(point, point1):  # instead of math.dist. since python3.5 on CloudVeneto doesn't support it
    d = 0
    for i in range(len(point)):
        d += (point[i] - point1[i]) ** 2
    return d


def calc_sp(point, cluster_id, clusteringSample):
    a = 0
    b = []

    for cluster_coreset_each in clusteringSample:
        distance_sum_to_coreset = 0
        for point_ in cluster_coreset_each[1]:
            distance_sum_to_coreset += math_dist(point, point_)**2
        if cluster_id == cluster_coreset_each[0]:
            a = distance_sum_to_coreset/min(M, currentClusteringSizes.value.get(cluster_coreset_each[0]))
        else:
            b.append(distance_sum_to_coreset/min(M, currentClusteringSizes.value.get(cluster_coreset_each[0])))
    b = min(b)
    # print(
    #     T,
    #     sharedClusterSizes.value.get(cluster_id),
    #     point,
    #     cluster_id,
    #     a,
    #     b,
    #     abs(b - a) / max(a, b)
    # )

    return abs(b - a) / max(a, b)


conf = SparkConf().setAppName('Clustering').setMaster("local[*]")
# conf = SparkConf().setAppName('Homework3').set('spark.locality.wait', '0s')
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

time_start_reading = time.time()
inputPoints = sc.textFile(data_path)\
    .cache()\
    .map(strToTuple)\
    .repartition(numPartitions=L)
# print(inputPoints.take(5))
print('Time for input reading = ', int((time.time() - time_start_reading)*1000))


for k in range(kstart, kstart+h):
    time_start_clustering = time.time()
    kmean_model = KMeans.train(inputPoints, k, maxIterations=iter, initializationMode="random")
    time_end_clustering = time.time()

    currentClustering = inputPoints.map(lambda row: (row, kmean_model.predict(row)))
    # print('currentClustering: ', currentClustering.take(5))

    currentClusteringSizes = sc.broadcast(currentClustering.values().countByValue())
    # print(currentClusteringSizes.value)

    M = int(m/k)

    clusteringSample = currentClustering.filter(
        lambda x: np.random.uniform(0.0, 1.0) < min(M / currentClusteringSizes.value.get(x[1]), 1)
    )
    # print('clusteringSample: ', clusteringSample.take(5))

    clusteringSampleSizes = sc.broadcast(clusteringSample.values().countByValue())
    # print(clusteringSampleSizes.value)

    clusteringSample_broadcast = sc.broadcast(
        clusteringSample
            .map(lambda x: (x[1], x[0]))
            .groupByKey()
            .collect()
    )
    # print(clusteringSample_broadcast.value)

    time_start_approxSilhFull = time.time()
    approxSilhFull = currentClustering \
        .map(
            lambda cluster_point: calc_sp(
                cluster_point[0],
                cluster_point[1],
                clusteringSample_broadcast.value
            )
        ) \
        .sum() / currentClustering.count()
    time_end_approxSilhFull = time.time()

    print()
    print('Number of clusters k = ', k)
    print('Silhouette coefficient = ', approxSilhFull)
    print('Time for clustering = ', int((time_end_clustering - time_start_clustering)*1000))
    print('Time for silhouette computation = ', int((time_end_approxSilhFull - time_start_approxSilhFull)*1000))
