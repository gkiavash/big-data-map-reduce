from pyspark import SparkContext, SparkConf
import sys
import os

assert len(sys.argv) == 4, "Usage: python WordCountExample.py <K> <T> <file_name>"

conf = SparkConf().setAppName('WordCountExample').setMaster("local[*]")
sc = SparkContext(conf=conf)

K = sys.argv[1]
assert K.isdigit(), "K must be an integer"
K = int(K)

T = sys.argv[2]
assert T.isdigit(), "T must be an integer"
T = int(T)

data_path = sys.argv[3]
assert os.path.isfile(data_path), "File or folder not found"
docs = sc.textFile(data_path, minPartitions=K).cache()
docs.repartition(numPartitions=K)


# final = docs\
#     .map(lambda line: (line.split(',')[1], (line.split(',')[0], line.split(',')[2])))\
#     .join(user_ave)\
#     .mapValues(lambda x: (x[0][0], float(x[0][1]) - float(x[1])))\
#     .values()\
#     .map(lambda movie_rate: (movie_rate[0], (float(movie_rate[1]), 1)))\
#     .reduceByKey(avg_reduce_func)\
#     .mapValues(lambda x: x[0]/x[1])\
#     .sortBy(lambda a: a[1], ascending=False)
# print(final.take(10))

def f_arr(arr):
    arr = list(arr)
    if len(arr) == 0:
        return []

    sum = 0
    for i in arr:
        sum += i[0]
    user_ave_ = sum / len(arr)
    return [(i, user_ave_) for i in arr]


RawData2 = docs\
    .map(lambda row:
                 (
                     row.split(',')[1],  # UserID
                     (
                         float(row.split(',')[2]),  # Rate
                         row.split(',')[0],  # ProductID
                         # 1,  # count
                     )
                 )
         )\
    .groupByKey()\
    .mapValues(f_arr)\
    .values()\
    .flatMap(lambda row: row)

normalizedRatings2 = RawData2\
    .map(lambda x: (x[0][1], x[0][0] - x[1]))

maxNormRatings2 = normalizedRatings2\
    .reduceByKey(lambda x, y: max(x, y))\
    .sortBy(lambda a: a[1], ascending=False)

print('INPUT PARAMETERS: K={} T={} file={}'.format(K, T, data_path))

for result in maxNormRatings2.take(T):
    print("Product {} maxNormRating {} ".format(result[0], result[1]))
