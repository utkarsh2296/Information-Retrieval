import timeit
import joblib
import math
import time
from nltk import PorterStemmer, re
from numpy.core import unicode
import json
import matplotlib.pyplot as plt

# final_dict = joblib.load('dataDictionary.pkl')  #this is sorted

# count = 0
# for key in final_dict.keys():
#     # final_dict[key] = sorted(final_dict[key], key=lambda expression: re.sub('[^A-Za-z]+', '', expression))
#     final_dict[key] = sorted(final_dict[key])
# joblib.dump(final_dict, "dataDictionary.pkl")
with open('dataDictionary2.json') as json_data:
    final_dict=json.load(json_data)
# print(final_dict.keys())

def preprocess(word):
    ps = PorterStemmer()
    return final_dict[ps.stem(word)]

def orFunction(list_word1, list_word2):
    # print(len(list_word1))
    # print(len(list_word2))

    start_time = time.time()
    result = list(set(list_word1 + list_word2))
    end_time = time.time()

    print("Time taken by or :  {0} microseconds".format((end_time - start_time) * 1000000))

    return result

def andFunction(list_word1, list_word2):
    len1 = len(list_word1)
    len2 = len(list_word2)

    result = []
    i, j = 0, 0
    iter_count = 0
    # start_time = time.time()
    start_time = timeit.default_timer()
    while(i<len1 and j<len2):
        if (list_word1[i]) == (list_word2[j]):
            result.append(list_word1[i])
            i += 1
            j += 1
            iter_count += 1

        elif (list_word1[i]) < (list_word2[j]):
            i += 1
            iter_count += 1
        else:
            j += 1
            iter_count += 1

    # end_time = time.time()
    end_time = timeit.default_timer()
    print("iter_count by AND: >>>", iter_count)
    time_taken = (end_time - start_time) * 1000000
    print("Time taken by AND:  {0} microseconds".format(time_taken))

    return result

def andNot(list_word1, list_word2):
    len1 = len(list_word1)
    len2 = len(list_word2)

    result = []
    index = len1

    i,j = 0, 0
    start_time = time.time()

    for k in range(index):
        if list_word1[i] == list_word2[j]:
            # print("yes")
            i += 1
            j += 1
            continue
        else:
            result.append(list_word1[i])

        if list_word1[i]<list_word2[j]:
            i += 1
        else:
            j += 1
    end_time = time.time()
    time_taken = (end_time - start_time) * 1000000
    print("Time taken by AND-NOT:  {0} microseconds".format(time_taken))

    return result

def orNot(list_word1, list_word2):
    len1 = len(list_word1)
    len2 = len(list_word2)

    with open('all_names2.json') as json_data:
        complete_list = json.load(json_data)

    start_time = time.time()
    for x in list_word2 :
        complete_list.remove(x)
    end_time = time.time()

    time_taken = (end_time - start_time) * 1000000
    print("Time taken by OR-NOT:  {0} microseconds".format(time_taken))
    return orFunction(list_word1, complete_list)

def skipPointer(list_word1, list_word2, factor = 1):
    len1 = len(list_word1)
    len2 = len(list_word2)

    pos1 = int(math.floor(math.sqrt(len(list_word1))) * factor)
    pos2 = int(math.floor(math.sqrt(len(list_word2))) * factor)
    # print("pos1 ", pos1)
    # print("pos2 ", pos2)

    result = []
    i, j = 0, 0
    iter_count = 0

    # start_time = time.time()
    start_time = timeit.default_timer()
    while (i < len1 and j < len2):
        # print(i)
        if (list_word1[i]) == (list_word2[j]):
            result.append(list_word1[i])
            i += 1
            j += 1
            iter_count += 1

        elif (list_word1[i]) < (list_word2[j]):
            iter_count += 1
            # print("yes")
            # if i+pos1 < len1 and list_word1[i+pos1] <= list_word2[j]:
                    # print("yes3")
            if i%pos1 == 0 and (int(i/pos1)+1)*pos1 <len1 and list_word1[(int(i/pos1)+1)*pos1] < list_word2[j]:
                    i = (int(i/pos1)+1)*pos1
            else:
                i += 1

        else:
            if j%pos2 == 0 and (int(j/pos2)+1)*pos2 <len2 and list_word2[(int(j/pos2)+1)*pos2] < list_word1[i]:
                    j = (int(j/pos2)+1)*pos2
            else:
                j += 1
    # end_time = time.time()
    end_time = timeit.default_timer()
    print("iter_count by skip method AND>>>", iter_count)
    t = (end_time - start_time)*1000000
    print("Time taken by skip method AND:  {0} microseconds".format(t))

    #comparing with AND
    andFunction(list_word1, list_word2)

    return result, iter_count, t

test1 = ['test', 'grate', 'problem', 'need']
test2 = ['pro', 'hit', 'news', 'thing']


def plot(test1, test2):
    iter_count = []
    time = []
    factor = [0.5, 1, 2, 3, 4, 5]
    for i in range(len(test1)):
        l1 = preprocess(test1[i])
        l2 = preprocess(test2[i])
        print("------------------------------------------")
        print("word1 : ", test1[i], "       word2 : ", test2[i])
        print("------------------------------------------")
        temp_iter_count = []
        temp_time = []
        print("yes1")
        for j in range(len(factor)):
            print("factor : ", factor[j])
            r, it, t = skipPointer(l1, l2, factor=factor[j])
            print(t)
            temp_iter_count.append(it)
            print("haha")
            temp_time.append(t)
        iter_count.append(temp_iter_count)
        time.append(temp_time)

        # plt.plot(iter_count[i], label=test1[i])
        # plt.plot(time[i], label=test2[i])



    for i in range(len(test1)):
        plt.plot(iter_count[i], label = test1[i] + "AND" + test2[i] )
        # plt.plot(time[i],  label = test1[i] + " AND " + test2[i])
        plt.xlabel("skip pointer length")
        plt.ylabel("time(micro seconds")
        plt.title("graph showing time in skip method")
    plt.legend()
    plt.show()

# plot(test1, test2)
