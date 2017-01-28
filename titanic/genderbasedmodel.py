import csv as csv
import numpy as np 

csv_file_object = csv.reader(open('train.csv', 'rb'))

header = csv_file_object.next()

data = []

for row in csv_file_object:
    data.append(row)

data = np.array(data)

percentage_total_who_survived = np.sum(data[0::, 1].astype(np.float)) / np.size(data[0::, 1].astype(np.float))

women_only_stats = data[0::, 3] == 'female'
men_only_stats = data[0::, 3] != 'female'

percentage_women_who_survived = np.sum(data[women_only_stats, 1].astype(np.float)) / np.size(data[women_only_stats, 1].astype(np.float)) 

test_file = open('test.csv', 'rb')

test_file_object = csv.reader(test_file)

header = test_file_object.next()

prediction_file = open('genderbasedpredictions.csv', 'wb')

prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(['PassengerID', 'Survived'])

for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], 1])
    else:
        prediction_file_object.writerow([row[0], 0])

test_file.close()

prediction_file.close()

