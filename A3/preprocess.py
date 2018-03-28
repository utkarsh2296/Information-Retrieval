import os
import string
import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords


class TfIdf:
    def __init__(self, path):
        self.path = path
        self.feature_selection_words = []

        self.matrix = {}
        # if os.path.isfile('file_map.pkl'):
        #     self.files = joblib.load('file_map.pkl')
        # else:
        #     self.files = self.directories()
        self.files = {}
        print("running")
        print("calculating tf-idf matrix...")

    def run(self, files):
        self.files = files
        for key in files.keys():
            self.readFile(files[key], key)

        self.idf_update()
        print("matrix calculation done!!")
        # self.save()

    def readFile(self, content, label):
        # file1 = open(file_path, encoding='latin')
        # content = file1.read()
        # stemmed_sentence = content.split('\n')
        words, counts = np.unique(content, return_counts=True)

        column_with_label = label
        # print("column : ", column)

        self.tf_update(words, counts, column_with_label)

        return

    def get_files(self, path):
        files = []
        classes = []

        for file in os.scandir(path):
            if file.is_file():
                # print("file : ", file)
                files.append(file.path)
                classes.append(os.path.basename(path))
            else:
                # print(file.path)
                ret_files, ret_classes = self.get_files(file.path)
                files.extend(ret_files)
                classes.extend(ret_classes)

        return files, classes

    def directories(self):
        print("here in directories")
        files, _ = self.get_files(self.path)
        # print("total files : ", len(files))
        files, numbers = np.unique(files, return_inverse=True)
        dict_files = dict(zip(files, numbers))
        return dict_files

    def tf_update(self, words, counts, column):
        count = 0

        for word in words:
            tf = (1 + np.math.log(counts[count]))

            if word in self.matrix.keys():
                row = self.matrix[word]
                row[column] = tf
                self.matrix[word] = row
                count += 1
            else:
                row = [0] * (len(self.files) + 1)
                row[column] = tf
                self.matrix[word] = row
                count += 1

    def idf_update(self):
        print("here")
        for key in self.matrix.keys():
            if key != "column_label":
                count = np.count_nonzero(self.matrix[key])
                idf = float(np.math.log(len(self.files) / count))
                self.matrix[key][-1] = idf
                # self.matrix[key] = np.multiply(self.matrix[key], idf)
        print("done")

    def preprocess(self, content):
        content = content.lower()

        # remove header and footer
        content = content.split('\n')

        start_count = 0
        for line in content:
            line = line.strip()
            if line == '':
                break
            else:
                start_count += 1

        content = content[start_count:]
        content = [line for line in content if not line.strip().endswith(":")]
        content = (' '.join(content))
        punctuation_remover = str.maketrans('', '', string.punctuation)
        content = content.translate(punctuation_remover)

        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokenized_content = tokenizer.tokenize(content)

        stop_words = set(stopwords.words("english"))
        filtered_sentence = [token for token in tokenized_content if not token in stop_words]

        ps = nltk.PorterStemmer()
        stemmed_sentence = [ps.stem(word) for word in filtered_sentence]

        return stemmed_sentence

    def feature_selection(self, percent):
        print("feature selection started")
        key_words = list(self.matrix.keys())
        # print("keywords : ", key_words)

        # print("col : ",)
        for col in range(len(self.files)):
            # if col%100 == 0:
                # print(col)
            tf_with_idf = []
            for key in self.matrix.keys():
                if key != "column_label":
                    # print("value : ", self.matrix[key][col], ", ", self.matrix[key][-1])
                    tf_with_idf.append(self.matrix[key][col] * self.matrix[key][-1])
            tf_with_idf = np.array(tf_with_idf)

            required_values = int(percent * np.count_nonzero(tf_with_idf))
            top_indexes = tf_with_idf.argsort()
            top_indexes = top_indexes[::-1]
            top_indexes = top_indexes[:required_values]

            self.feature_selection_words += [key_words[index] for index in top_indexes]
        self.feature_selection_words = list(np.unique(self.feature_selection_words))
        joblib.dump(self.feature_selection_words, "feature.words")
        print("feature selection done")

    def save(self):
        joblib.dump(self.matrix, 'matrix.pkl')
        joblib.dump(self.files, 'file_map.pkl')
        print("...matrix calculation done!!saved!!")

