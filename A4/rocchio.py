import itertools
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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
    numbers = np.arange(len(files))
    dict_files = dict(zip(numbers, files))
    return dict_files


def read_file(file_path):
    # print(file_path)
    file = open(file_path, encoding='latin')
    content = file.read()
    content = content.split('\n')
    return content


inverted_dict = {}
doc_post_list = {}
centroids = {}
idf_values = {}


def create_inverted_index(files):
    for file_key in files.keys():
        words = read_file(files[file_key])
        words, counts = np.unique(words, return_counts=True)
        counts = [np.log(counts[index] + 1) for index in range(len(counts))]

        post_dict = dict(zip(words, counts))
        doc_post_list[file_key] = post_dict

        for index in range(len(words)):
            if words[index] in inverted_dict.keys():
                inverted_dict[words[index]].append((file_key, counts[index]))
            else:
                inverted_dict[words[index]] = []
                inverted_dict[words[index]].append((file_key, counts[index]))

    for key in inverted_dict.keys():
        idf = np.log(1 + (len(files) / len(inverted_dict[key])))
        idf_values[key] = idf


def create_centroids(x, y):
    classes, count = np.unique(y, return_counts=True)

    ct = 0
    for key in inverted_dict.keys():
        row = inverted_dict[key]

        class_doc_tfidf = [0] * 5
        for tup in row:
            file_index = tup[0]

            if y[file_index] == 0:
                class_doc_tfidf[0] += tup[1]
            elif y[file_index] == 1:
                class_doc_tfidf[1] += tup[1]
            elif y[file_index] == 2:
                class_doc_tfidf[2] += tup[1]
            elif y[file_index] == 3:
                class_doc_tfidf[3] += tup[1]
            else:
                class_doc_tfidf[4] += tup[1]

        word_list = [float(class_doc_tfidf[i] / count[i]) for i in range(len(classes))]
        centroids[key] = word_list
        ct += 1
        if ct % 1000 == 0:
            print(ct)


def predict(x):
    prediction = []
    for i in range(len(x)):
        words = read_file(x[i])
        words, counts = np.unique(words, return_counts=True)
        tf = [np.log(1 + count) for count in counts]

        tf_idf = []

        class0 = []
        class1 = []
        class2 = []
        class3 = []
        class4 = []

        for index in range(len(words)):
            if words[index] in (list(centroids.keys())):
                class0.append(centroids[words[index]][0] * idf_values[words[index]])
                class1.append(centroids[words[index]][1] * idf_values[words[index]])
                class2.append(centroids[words[index]][2] * idf_values[words[index]])
                class3.append(centroids[words[index]][3] * idf_values[words[index]])
                class4.append(centroids[words[index]][4] * idf_values[words[index]])
                tf_idf.append(tf[index] * idf_values[words[index]])
            else:
                class0.append(0)
                class1.append(0)
                class2.append(0)
                class3.append(0)
                class4.append(0)
                tf_idf.append(0)

        sim0 = cosine_similarity(class0, tf_idf)
        sim1 = cosine_similarity(class1, tf_idf)
        sim2 = cosine_similarity(class2, tf_idf)
        sim3 = cosine_similarity(class3, tf_idf)
        sim4 = cosine_similarity(class4, tf_idf)

        result = np.argmax([sim0, sim1, sim2, sim3, sim4])
        prediction.append(result)
    return prediction


def predictKnn(x):
    doc_result = {}
    for i in range(len(x)):
        print(i)
        words = read_file(x[i])
        words, counts = np.unique(words, return_counts=True)
        tf = [np.log(1 + count) for count in counts]
        tf_idf = []
        for index in range(len(tf)):
            if words[index] in idf_values.keys():
                tf_idf.append(tf[index] * idf_values[words[index]])
            else:
                tf_idf.append(tf[index])

        all = []
        for doc_key in doc_post_list.keys():
            doc_vec = []
            # print("ha", doc_key)
            for word in words:
                # print(word, "hahaha1")
                if word in doc_post_list[doc_key].keys():
                    # print(word, "hahaha")
                    val = doc_post_list[doc_key][word]
                    idf = idf_values[word]
                    doc_vec.append(val * idf)
                else:
                    doc_vec.append(0)

            cos_sim = cosine_similarity(tf_idf, doc_vec)
            all.append(cos_sim)

        doc_result[i] = all

    return doc_result


def get_k_output(y, k, doc_result):
    prediction = []
    print("len y : ", len(y))

    for key in doc_result.keys():
        sorted_index = np.argsort(doc_result[key])
        output_k = sorted_index[-k:]
        out = y[output_k]
        o, c = np.unique(out, return_counts=True)
        prediction.append(o[np.argmax(c)])

    return prediction


def cosine_similarity(doc1, doc2):
    div1 = np.linalg.norm(doc1)
    div2 = np.linalg.norm(doc2)

    val = 0

    for i in range(len(doc1)):
        val += (doc1[i] * doc2[i])

    return float(val / (div1 * div2))


def accuracy(actual, predicted):
    count = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1

    return float(count / len(actual))

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



split = [0.53, 0.23, 0.13]
for s in split:
    x, y = get_files("F:\IIITD\Semester_2\Information-Retrieval\A3\data_processed")
    un, y = np.unique(y, return_inverse=True)
    print(un)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=s, stratify=list(y), shuffle=True, random_state=5)
    files_dict = directories(train_x)
    create_inverted_index(files_dict)
    create_centroids(train_x, train_y)
    r = predict(test_x)
    print("rochhio acc : ", str(s)," : ", accuracy(r, test_y))

    cnf_matrix = confusion_matrix(r, test_y)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(test_y),
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(test_y), normalize=True,
                          title='Normalized confusion matrix')

    plt.savefig(str(s) + "_rocchio.jpg")

    r = predictKnn(test_x)

    k = [1,3,5]
    for ind in k:
        prediction = []
        for key in r.keys():
            sorted_index = np.argsort(r[key])
            output_k = sorted_index[-ind:]

            out = train_y[output_k]
            o, c = np.unique(out, return_counts=True)
            prediction.append(o[np.argmax(c)])

        print("accuracy : ", str(s), " : k : ", str(ind), " : ", accuracy(prediction, test_y))
        cnf_matrix = confusion_matrix(prediction, test_y)

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=np.unique(test_y),
                              title='Confusion matrix, without normalization')
        plt.savefig(str(s) + "_" + str(k) + "_.jpg")
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=np.unique(test_y), normalize=True,
                              title='Normalized confusion matrix')

        plt.savefig(str(s) + "normalised_" + str(k) + "_.jpg")

        plt.show()

