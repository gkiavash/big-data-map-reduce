import csv

users = {}
whole_arr = []

FILE_NAME = 'input_20K.csv'


with open(FILE_NAME) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        if row[1] not in users.keys():
            users.update({row[1]: {'count': 1,
                                   'sum': float(row[2])}})
        else:
            users.update({
                row[1]: {
                    'count': users.get(row[1]).get('count')+1,
                    'sum': users.get(row[1]).get('sum')+float(row[2])
                }
            })
        whole_arr.append(row)

for user in users.keys():
    print(user)
    user_ = users.get(user)
    users.get(user).update({'ave': float(user_.get('sum'))/float(user_.get('count'))})
print(users)


movies = {}
with open(FILE_NAME) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if row[0] not in movies.keys():
            movies.update(
                {
                    row[0]: {
                        'count': 1,
                        'sum': float(row[2])-users.get(row[1]).get('ave'),
                        'max': float(row[2])-users.get(row[1]).get('ave')
                    }
                }
            )
        else:
            movies.update(
                {
                    row[0]: {
                        'count': movies.get(row[0]).get('count')+1,
                        'sum': movies.get(row[0]).get('sum') + float(row[2]) - users.get(row[1]).get('ave'),
                        'max': max(movies.get(row[0]).get('max'), float(row[2]) - users.get(row[1]).get('ave'))
                    }
                }
            )

for movie in movies.keys():
    print(movie)
    movie_ = movies.get(movie)
    movies.get(movie).update({'ave': float(movie_.get('sum'))/float(movie_.get('count'))})

print(len(movies))
# print(movies)

a = {
    'A': {'ave': 1},
    'B': {'ave': 3},
    'C': {'ave': 2},
}
movies_ = dict(sorted(movies.items(), key=lambda item: item[1].get('ave')))
print(movies_)
movies_ = dict(sorted(movies.items(), key=lambda item: item[1].get('max')))
print(movies_)
