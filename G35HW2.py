from pyspark import SparkContext, SparkConf
import random
import sys
import os
import numpy as np
import time
import math

assert len(sys.argv) == 4, "Usage: python G35HW2.py <file_name> <K> <T>"

conf = SparkConf().setAppName('Clustering').setMaster("local[*]")
sc = SparkContext(conf=conf)

K = sys.argv[2]
assert K.isdigit(), "K must be an integer"
K = int(K)

T = sys.argv[3]
assert T.isdigit(), "T must be an integer"
T = int(T)

if T > 1000:
    p = 8
else:
    p = 4


def calc_sp(point, cluster_id, clusteringSample):
    a = 0
    b = []

    for cluster_coreset_each in clusteringSample:
        distance_sum_to_coreset = 0
        for point_ in cluster_coreset_each[1]:
            distance_sum_to_coreset += math.dist(point, point_)**2
        if cluster_id == cluster_coreset_each[0]:
            a = distance_sum_to_coreset/min(T, sharedClusterSizes.value.get(cluster_coreset_each[0]))
        else:
            b.append(distance_sum_to_coreset/min(T, sharedClusterSizes.value.get(cluster_coreset_each[0])))
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


def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch) - 1))
    return (point, int(ch[-1]))  # returns (point, cluster_index)


data_path = sys.argv[1]
assert os.path.isfile(data_path), "File or folder not found"

fullClustering = sc.textFile(data_path)\
    .cache()\
    .map(strToTuple)\
    .repartition(numPartitions=p)
# print(fullClustering.take(5))

sharedClusterSizes = sc.broadcast(fullClustering.values().countByValue())
# print(sharedClusterSizes.value)

clusteringSample = fullClustering.filter(lambda x: np.random.uniform(0.0, 1.0) < min(T / sharedClusterSizes.value.get(x[1]), 1))
# print(clusteringSample.take(5))

clusteringSampleSizes = sc.broadcast(clusteringSample.values().countByValue())
# print(clusteringSampleSizes.value)

clusteringSample_broadcast = sc.broadcast(
    clusteringSample
        .map(lambda x: (x[1], x[0]))
        .groupByKey()
        .collect()
)
# print(clusteringSample_broadcast.value)

start_time = time.time()
approxSilhFull = fullClustering\
    .map(
        lambda cluster_point: calc_sp(
            cluster_point[0],
            cluster_point[1],
            clusteringSample_broadcast.value
        )
    )\
    .sum()/fullClustering.count()
end_time = time.time()
print('Value of approxSilhFull = ', approxSilhFull)
print("Time to compute approxSilhFull = {} ms".format(int((end_time - start_time)*1000)))

start_time = time.time()
exactSilhSample = clusteringSample\
    .map(
        lambda cluster_point: calc_sp(
            cluster_point[0],
            cluster_point[1],
            clusteringSample_broadcast.value
        )
    )\
    .sum()/clusteringSample.count()
end_time = time.time()
print('Value of exactSilhSample = ', exactSilhSample)
print("Time to compute exactSilhSample = {} ms".format(int((end_time - start_time)*1000)))
