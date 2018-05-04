# import itertools
# import matplotlib.pyplot as plt
# import numpy as np
# #
# # rocchio = [0.905, 0.91, 0.92]
# #
# # knn
# #
# # accuracy :  0.1  : k :  1  :  0.928
# # accuracy :  0.1  : k :  3  :  0.926
# # accuracy :  0.1  : k :  5  :  0.9148
# #
# # accuracy :  0.2  : k :  1  :  0.923
# # accuracy :  0.2  : k :  3  :  0.909
# # accuracy :  0.2  : k :  5  :  0.903
# #
# # accuracy :  0.5  : k :  1  :  0.8928
# # accuracy :  0.5  : k :  3  :  0.8624
# # accuracy :  0.5  : k :  5  :  0.8416
#
# # naive bayes
#
# rocchio = [0.905, 0.91, 0.92]
# nb = [0.9376, 0.952,  0.94]
# knn_1 = [0.928, 0.926, 0.914]
# knn_3 = [0.923, 0.909, 0.903]
# knn_5 = [0.8928, 0.8624, 0.8416]
#
#
# plt.plot(nb, label="naive bayes")
# plt.plot(rocchio, label="rocchio")
# plt.plot(knn_1, label="knn with k=1")
# plt.plot(knn_3, label="knn with k=3")
# plt.plot(knn_5, label="knn with k=5")
# plt.title("comarision of classification methods at different split")
# plt.xlabel("split ratios")
# plt.xticks(np.arange(3), [0.5, 0.2, 0.1])
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import matplotlib.pyplot as plt
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
#
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=True,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues, name = "matrix.jpg"):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure()
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         print("i", i)
#         print("j", j)
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig(name)
#
# classes_n = ['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']
# classes = [0,1,2,3,4]
#
# rcc_05 = np.array([[0.94157303, 0.00224719, 0.02022472, 0.03370787, 0.00224719]
# , [0.00838574, 0.9769392, 0.00838574, 0.00419287, 0.00209644]
# , [0.03030303, 0.00808081, 0.92727273, 0.02424242, 0.01010101]
# , [0.0952381, 0.00595238, 0.01785714, 0.86309524, 0.01785714]
# , [0.02417962, 0.04490501, 0.0328152, 0.06217617, 0.83592401]])
# rcc_02 = np.array([[0.90909091, 0.00505051, 0.02525253, 0.06060606, 0.]
# , [0.00526316, 0.97894737, 0.00526316, 0.01052632, 0.]
# , [0.01587302, 0.01587302, 0.95238095, 0.01058201, 0.00529101]
# , [0.05208333, 0.015625, 0.02083333, 0.90104167, 0.01041667]
# , [0.02597403, 0.03030303, 0.04329004, 0.04761905, 0.85281385]])
# rcc_01 = np.array([[0.873, 0.0097, 0.0388, 0.06796117, 0.00970874]
# , [0., 0.95918367, 0.02040816, 0., 0.02040816]
# , [0.03370787, 0., 0.96629213, 0., 0.]
# , [0.04, 0.02, 0.04, 0.9, 0.]
# , [0.02727273, 0.02727273, 0.03636364, 0.02727273, 0.88181818]])
#
# knn_05_1 = np.array([[0.91044776, 0.0021322, 0.04690832, 0.02771855, 0.01279318]
# , [0.03454545, 0.88, 0.03272727, 0.03818182, 0.01454545]
# , [0.03023758, 0.00215983, 0.91792657, 0.02159827, 0.02807775]
# , [0.06490872, 0.01014199, 0.03042596, 0.87626775, 0.01825558]
# , [0.0152381, 0.01714286, 0.03809524, 0.04571429, 0.88380952]])
# knn_05_3 = np.array([[0.79268293, 0.0174216, 0.10452962, 0.06620209, 0.01916376]
# , [0.02300885, 0.83185841, 0.04778761, 0.05132743, 0.0460177, ]
# , [0.01428571, 0.00952381, 0.91428571, 0.03333333, 0.02857143]
# , [0.04954955, 0.01576577, 0.02027027, 0.90315315, 0.01126126]
# , [0.00804829, 0.01810865, 0.04024145, 0.0362173, 0.89738431]])
# knn_05_5 = np.array([[0.68445122, 0.03506098, 0.13414634, 0.11890244, 0.02743902]
# , [0.01473684, 0.94526316, 0.01263158, 0.01473684, 0.01263158]
# , [0.03881279, 0.01141553, 0.86073059, 0.03652968, 0.05251142]
# , [0.04672897, 0.01401869, 0.02803738, 0.89485981, 0.01635514]
# , [0.0139165, 0.03379722, 0.03379722, 0.03180915, 0.88667992]])
#
# knn_02_1 = np.array([[0.92039801, 0.00497512, 0.01492537, 0.05472637, 0.00497512]
# , [0.01, 0.95, 0.01, 0.015, 0.015]
# , [0.01554404, 0.00518135, 0.94300518, 0.01554404, 0.02072539]
# , [0.02955665, 0.01477833, 0.04926108, 0.8817734, 0.02463054]
# , [0.01970443, 0.02463054, 0.01477833, 0.01970443, 0.92118227]])
# knn_02_3 = np.array([[0.80932203, 0.01694915, 0.0720339, 0.08474576, 0.01694915]
# , [0.00490196, 0.93137255, 0.00980392, 0.02941176, 0.0245098]
# , [0.01639344, 0., 0.94535519, 0.01092896, 0.0273224]
# , [0.02197802, 0.01648352, 0.02197802, 0.93406593, 0.00549451]
# , [0.00512821, 0.01538462, 0.02051282, 0.01025641, 0.94871795]])
# knn_02_5 = np.array([[0.80508475, 0.02118644, 0.07627119, 0.09322034, 0.00423729]
# , [0.01530612, 0.94897959, 0.00510204, 0.01530612, 0.01530612]
# , [0.00546448, 0., 0.94535519, 0.02185792, 0.0273224]
# , [0.01117318, 0.02234637, 0.02234637, 0.9273743, 0.01675978]
# , [0.01941748, 0.02427184, 0.01941748, 0.02427184, 0.91262136]])
#
# knn_01_1 = np.array([[0.88785047, 0.00934579, 0.03738318, 0.05607477, 0.00934579]
# , [0.02020202, 0.94949495, 0.01010101, 0., 0.02020202]
# , [0.01086957, 0., 0.9673913, 0., 0.02173913]
# , [0.01941748, 0.00970874, 0.04854369, 0.90291262, 0.01941748]
# , [0., 0.04040404, 0.01010101, 0.01010101, 0.93939394]])
# knn_01_3 = np.array([[0.8220339, 0.00847458, 0.05932203, 0.09322034, 0.01694915]
# , [0., 0.96969697, 0., 0., 0.03030303]
# , [0.01075269, 0., 0.95698925, 0.01075269, 0.02150538]
# , [0.01075269, 0.02150538, 0.02150538, 0.94623656, 0.]
# , [0.01030928, 0.01030928, 0.02061856, 0., 0.95876289]])
# knn_01_5 = np.array([[0.78225806, 0.00806452, 0.06451613, 0.12903226, 0.01612903]
# , [0.02020202, 0.96969697, 0., 0., 0.01010101]
# , [0., 0., 0.97752809, 0., 0.02247191]
# , [0.01123596, 0.02247191, 0.02247191, 0.93258427, 0.01123596]
# , [0., 0.01010101, 0.03030303, 0.01010101, 0.94949495]])
#
# plot_confusion_matrix(rcc_05, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="rcc_05.jpg")
# plot_confusion_matrix(rcc_02, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="rcc_02.jpg")
# plot_confusion_matrix(rcc_01, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="rcc_01.jpg")
#
# plot_confusion_matrix(knn_05_1, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_05_1.jpg")
# plot_confusion_matrix(knn_05_3, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_05_3.jpg")
# plot_confusion_matrix(knn_05_5, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_05_5.jpg")
#
# plot_confusion_matrix(knn_02_1, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_02_1.jpg")
# plot_confusion_matrix(knn_02_3, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_02_3.jpg")
# plot_confusion_matrix(knn_02_5, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_02_5.jpg")
#
# plot_confusion_matrix(knn_01_1, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_01_1.jpg")
# plot_confusion_matrix(knn_01_3, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_01_3.jpg")
# plot_confusion_matrix(knn_01_5, classes=classes, title='Confusion matrix for rochhio at 50:50 split', name="knn_01_5.jpg")


import matplotlib.pyplot as plt

plt.plot([1,2,3])
plt.title(" 80:20 split       90:10 split")
plt.show()