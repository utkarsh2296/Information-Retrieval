import os
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.externals import joblib

from preprocess import TfIdf


class NaiveBayes:
    def __init__(self, n_class):
        self.n_class = n_class
        self.conditional_prob_matrix = []

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

    def fit(self, train_files, train_labels, feature_select=False, feature_percent=0.3):
        print("training...")
        feature_selected_words = None
        train_files_content = self.create_files(train_files, train_labels)
        if feature_select:

            print("-----------Applying Feature selection----------")
            ti = TfIdf("data_preprocessed")
            ti.run(train_files_content)
            ti.feature_selection(percent=feature_percent)
            # feature_selected_words = joblib.load('feature.words')
            feature_selected_words = ti.feature_selection_words
            feature_selected_words.append('my_unknown')
            # print(feature_selected_words)
        else:
            print("-----------Applying without Feature selection----------")

        for label in range(self.n_class):
            words = []
            for index in train_files_content.keys():
                if index == label:
                    # file = open(train_files[index], encoding='latin')
                    # content = file.read()
                    # stemmed_sentence = content.split('\n')
                    stemmed_sentence = train_files_content[index]
                    # content = content.split(" ")
                    # print("length : ", len(content))
                    # stemmed_sentence = self.preprocess(content)
                    words += stemmed_sentence

            words, counts = np.unique(words, return_counts=True)
            words_n = []
            counts_n = []

            print("length before : ", len(words))
            if feature_select:
                feature_words_class = feature_selected_words[:]
                for i in range(len(words)):
                    if words[i] in feature_words_class:
                        words_n.append(words[i])
                        counts_n.append(counts[i])
            else:
                words_n = words[:]
                counts_n = counts[:]

            # print("length after : ", len(words_n))
            prob = self.get_probability(counts_n)
            words_n = list(words_n)
            words_n.append('my_unknown')
            words_prob_dict = dict(zip(words_n, prob))
            self.conditional_prob_matrix.append(words_prob_dict)
            # print("done class : ", label)

    def predict(self, test_files, n_class=5):
        prediction = []
        for file_name in test_files:
            cond_prob = []
            file = open(file_name, encoding='latin')
            content = file.read()
            stemmed_sentence = content.split('\n')
            stemmed_sentence = list(set(stemmed_sentence))

            for label in range(n_class):
                prob = 0
                for word in stemmed_sentence:
                    if word in self.conditional_prob_matrix[label].keys():
                        prob += self.conditional_prob_matrix[label][word]
                    else:
                        prob += self.conditional_prob_matrix[label]['my_unknown']

                cond_prob.append(prob)

            prediction.append(np.argmax(cond_prob))

        return prediction

    def score(self, test_files, test_labels):
        predicted = self.predict(test_files)

        c = 0
        for i in range(len(predicted)):
            if predicted[i] == test_labels[i]:
                c += 1
        print(float(c / len(predicted)))

        return float(c / len(predicted))

    def get_probability(self, counts):
        prob = []
        for i in counts:
            p = float(np.log((i + 1) / (np.sum(counts) + len(counts) + 1)))
            prob.append(p)
        prob.append(float(np.log(1 / (np.sum(counts) + len(counts) + 1))))

        return prob

    def create_files(self, files, labels):
        dict = {}

        for index in range(len(files)):
            f = open(files[index])
            content = f.read()
            content = content.split('\n')
            if labels[index] in dict.keys():
                dict[labels[index]] += content
            else:
                dict[labels[index]] = content

        return dict
