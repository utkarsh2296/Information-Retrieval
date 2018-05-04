import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


class tf_idf_matrix:
    def __init__(self, files_index):
        self.path = "F:\IIITD\Semester_2\Information-Retrieval\A3\data_processed"
        print("here")
        self.matrix = {}
        self.files_index = files_index

        self.create_matrix()
        # print(self.files)

    def tf_calculate(self, words, counts, column):
        count = 0
        for word in words:
            tf = (1 + np.math.log10(counts[count]))

            if word in self.matrix.keys():
                row = self.matrix[word]
                row[0, column] = tf
                count += 1
                row[0, -1] += 1
                self.matrix[word] = row
            else:
                row = [0] * (self.files_index.__len__() + 1)
                row = lil_matrix(row)
                row[0, column] = tf
                row[0, -1] = 1
                self.matrix[word] = row

    def idf_update(self):
        print("len keys : ", len(self.matrix.keys()))
        for key in self.matrix.keys():
            # count = np.count_nonzero(self.matrix[key])
            count = self.matrix[key][0, -1]
            idf = float(np.log(self.files_index.__len__() / count))
            self.matrix[key][0, -1] = idf
            # row = self.matrix[key]
            # row = [row[0, i] *idf for i in range(row.shape[1])]
            # self.matrix[key] = row
        return

    def readFile(self, file_path):
        file = open(file_path, encoding='latin')
        content = file.read()
        content = content.split('\n')
        words, counts = np.unique(content, return_counts=True)

        return words, counts

    def create_matrix(self):
        print("inside create")
        count = 0
        print("create matrix func")
        for key in self.files_index.keys():
            words, counts = self.readFile(self.files_index[key])
            self.tf_calculate(words, counts, key)
            count += 1
            if count % 50 == 0:
                print(count)

        self.idf_update()
        # self.save()

def get_files(path):
    files = []
    classes = []

    for file in os.scandir(path):
        if file.is_file():
            files.append(file.path)
            classes.append(os.path.basename(path))
        else:
            ret_files, ret_classes = get_files(file.path)
            files.extend(ret_files)
            classes.extend(ret_classes)

    return files, classes


def directories(files):
    files, numbers = np.unique(files, return_inverse=True)
    dict_files = dict(zip(numbers, files))
    return dict_files


x, y = get_files("F:\IIITD\Semester_2\Information-Retrieval\A3\data_processed")
_, y = np.unique(y, return_inverse=True)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=20, stratify=y)
files_dict = directories(train_x)

m = tf_idf_matrix(files_dict)

centroids = {}

def create_centroids( x, y):
    classes, count = np.unique(y, return_counts=True)

    ct = 0
    for key in m.matrix.keys():
        row = m.matrix[key]
        word_list = []

        for c in range(len(classes)):
            class_doc_tfidf = 0
            for index in range(row.shape[1]):
                print("y index : ", y[index])
                if y[index] == c:
                    class_doc_tfidf += row[0, index]

            mean_value = float(class_doc_tfidf / count[c])
            word_list.append(mean_value)

        centroids[key] = word_list
        ct += 1
        if ct % 1000 == 0:
            print(ct)


create_centroids(train_x, train_y)











t = [1,2,3,0,4]
t2 = [10,20,30,0,40]
t = lil_matrix(t)
t2 = lil_matrix(t2)

r = {}
r[1] = t
r[2] = t2

xt = []
for i in r.values():
    xt.append(i.toarray()[0])

print(xt)
print(np.array(xt))
print()
xt = lil_matrix(xt)

xt = lil_matrix(list(r.values()))