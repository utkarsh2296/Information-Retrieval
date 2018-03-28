import glob, os, re
import json
import string
import joblib
import nltk
from nltk.corpus import stopwords

directory_path = '/home/corvo/Downloads/IR/Assignment1/20_newsgroups/20_newsgroups/'

def readFile(file_path, count):
    file = open(file_path, encoding='latin')
    content = file.readlines()
    # print(content.__len__())
    content = [i.strip() for i in content]
    # print("--------------------------------------------")
    # print(content)
    # print("---------------------------------------------")

    final_stemmed = []

    line_no = 0
    for line in content:
        # print(line_no)
        line_spaced = re.findall(r"[\w']+|[:]",line)

        if len(line_spaced) < 1:
            line_no+=1
            continue
        if line_spaced[0] != "Lines":
            line_no+=1
            continue
        else:
            # print("cool")
            break
    # print(line_no)
    counter = 0
    line_no = len(content) - line_no
    # print("line : ", line_no)

    for line in reversed(content):
        if counter == line_no:
            # print("yes")
            break
        counter += 1
        # new_content.append(line.lower())
        # punctuation_remover = str.maketrans('', '', string.punctuation)
        # line = line.translate(punctuation_remover)

        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokenized_content = tokenizer.tokenize(line)

        # punctuation_remover = str.maketrans('', '', string.punctuation)
        # line = line.translate(punctuation_remover)

        stop_words = set(stopwords.words("english"))
        filtered_sentence = [token for token in tokenized_content if not token in stop_words]

        # lemmed_sentence = [nltk.WordNetLemmatizer().lemmatize(word) for word in filtered_sentence]

        ps = nltk.PorterStemmer()

        stemmed_sentence = [ps.stem(word) for word in filtered_sentence]
        final_stemmed.extend(stemmed_sentence)
        # print(stemmed_sentence)

    return_final_stemmed = final_stemmed[:]
    final_stemmed = list(set(final_stemmed))
    # print(final_stemmed)

    #convert to dict
    # d = dict.fromkeys(final_stemmed, int(os.path.basename(file_path)))

    d = nltk.defaultdict(list)

    # file_path = file_path.split('/')
    # file_path = file_path[-2] + "/" + file_path[-1]
    file_path = str(count) + "_" + str(os.path.basename(file_path))

    for word in final_stemmed:
        d[word].append(file_path)
    return d, file_path, return_final_stemmed

# temp_path = '/home/corvo/Desktop/49960'
# _, _, word_set = readFile(temp_path, 1)
# print(word_set.__len__())


final_dict = {}
word_list = []
all_names = []

count = 0
for name in glob.glob(directory_path+"*"):
    if os.path.isdir(name):
        for inner_names in glob.glob(name + "/*"):
            # print(inner_names)
            dict, file_path, final_stemmed = readFile(inner_names, count)
            all_names.append(file_path)
            word_list.extend(final_stemmed)
            # print(dict)
            final_dict = nltk.defaultdict(list, final_dict)
            # print(final_dict)
            for i, j in dict.items():
                final_dict[i].extend(j)

            count += 1
            if count%1000 == 0:
               print(count)
    else:
        print(name)
print(count)


joblib.dump(word_list, 'word_list.pkl')


#sort the list of each key
for key in final_dict.keys():
    # final_dict[key] = sorted(final_dict[key], key=lambda expression: re.sub('[^A-Za-z]+', '', expression))
    final_dict[key] = sorted(final_dict[key])

# joblib.dump(all_names, 'all_names.json')
json_object = json.dump(all_names, open('all_names2.json', mode='w'))
# joblib.dump(final_dict, "final_dict.dump")
json_object = json.dump(final_dict, open('dataDictionary2.json', mode='w'))

'''
#load
final_dict = joblib.load('final_dict.dump')

for key in final_dict.keys():
    print(key),
    print(final_dict[key])

print(len(final_dict.keys()))
'''