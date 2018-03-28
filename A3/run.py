import os
import numpy as np
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes


def get_files(path):
    files = []
    classes = []

    for file in os.scandir(path):
        if file.is_file():
            # print("file : ", file.path)
            # f = open(file.path, encoding='latin')
            # content = f.read()
            # content = preprocess(content)
            # f_w = open(file.path, mode='w')
            # for word in content:
            #     f_w.write(word + "\n")
            # f_w.close()
            files.append(file.path)
            classes.append(os.path.basename(path))
        else:
            # print(file.path)
            ret_files, ret_classes = get_files(file.path)
            files.extend(ret_files)
            classes.extend(ret_classes)

    return files, classes


test_sizes = [0.1, 0.2, 0.3, 0.5]
feat_percent = [0.1, 0.3, 0.5, 0.7, 0.9]

accuracy = {}


# for t in test_sizes:
#     test_result = []
#     for f in feat_percent:
#         nb = NaiveBayes(5)
#         files, labels = get_files("data_processed")
#         x, _, labels = np.unique(labels, return_inverse=True, return_index=True)
#         train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=t,
#                                                                               stratify=labels,
#                                                                               random_state=7)
#
#         nb.fit(train_files, train_labels, feature_select=False, feature_percent=f)
#         print("score : test size : ", str(t))
#         nb.score(test_files, test_labels)
#         test_result.append(nb.score(test_files, test_labels))
#         pr = nb.predict(test_files)
#
#     accuracy[t] = test_result

accuracy = []


for t in test_sizes:
    test_result = []

    nb = NaiveBayes(5)
    files, labels = get_files("data_processed")
    x, _, labels = np.unique(labels, return_inverse=True, return_index=True)
    train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=t,
                                                                          stratify=labels,
                                                                          random_state=7)

    nb.fit(train_files, train_labels, feature_select=False, feature_percent=1)
    print("score : test size : ", str(t))
    nb.score(test_files, test_labels)
    test_result.append(nb.score(test_files, test_labels))
    pr = nb.predict(test_files)

    accuracy.append(test_result)

print(accuracy)

# test_sizes = [0.1, 0.2, 0.3, 0.5]
# feat_percent = [0.1, 0.3, 0.5, 0.7, 0.9]

# results = np.array([[0.664, 0.848, 0.91, 0.95, 0.948],
#            [0.617, 0.854, 0.924, 0.949, 0.956],
#            [0.534, 0.9146, 0.922, 0.95733, 0.9614],
#            [0.734, 0.9084, 0.894, 0.936, 0.9432]])
#
# for i in range(results.shape[1]):
#     plt.plot(results[:, i], label = "feature %age :" + str(feat_percent[i] * 100) + "%")
#
# plt.title("test accuracies against varying feature selection")
# plt.xlabel("split ratio")
# plt.ylabel("test accuracy")
# plt.xticks(np.arange(4), test_sizes)
# plt.title("varying feature selection for test splits")
# plt.legend()
# plt.show()
#
#
# x = [0.94, 0.952, 0.9534, 0.9376]
# plt.plot(x, label="normal without feature selection")
# plt.xticks(np.arange(4), test_sizes)
# plt.ylabel("test accuracy")
# plt.xlabel("test size ratio")
# plt.title("test accuracy normal without feature selection")
# plt.legend()
# plt.show()



