import sys
import os
import string
from nltk.stem import PorterStemmer

train_ham_files = []
train_spam_files = []
test_ham_files = []
test_spam_files = []

global stopwords_file

# fetch command line arguments
stopwords_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]

# declare variables required for processing NB
train_ham_dictionary = {}
train_spam_dictionary = {}
ps = PorterStemmer()
stopwords_list = []
total_no_of_docs_in_ham = 0
total_no_of_docs_in_spam = 0

def getpaths():

    global train_ham_files
    global train_spam_files
    global test_ham_files
    global test_spam_files
    global stopwords_file

    for root,dirs,files in os.walk(train_dir):
        for dir in dirs:
            if "ham" in dir.lower():
                directory_ham = os.path.join(train_dir, dir)
                for file in os.listdir(directory_ham):
                    if file.endswith(".txt"):
                        train_ham_files.append((os.path.join(directory_ham, file)))
        for dir in dirs:
            if "spam" in dir.lower():
                directory_spam = os.path.join(test_dir, dir)
                for file in os.listdir(directory_spam):
                    if file.endswith(".txt"):
                        train_spam_files.append((os.path.join(directory_spam, file)))
    for root, dirs, files in os.walk(test_dir):
        for dir in dirs:
            if "ham" in dir.lower():
                directory_ham = os.path.join(test_dir, dir)
                for file in os.listdir(directory_ham):
                    if file.endswith(".txt"):
                        test_ham_files.append((os.path.join(directory_ham, file)))
        for dir in dirs:
            if "spam" in dir.lower():
                directory_spam = os.path.join(test_dir, dir)
                for file in os.listdir(directory_spam):
                    if file.endswith(".txt"):
                        test_spam_files.append((os.path.join(directory_spam, file)))

    for root,dirs,files in os.walk(stopwords_dir):
        for file in files:
            if file.endswith('.txt'):
                stopwords_file = os.path.join(root,file)
                stopwords_file.replace("/", "\\")


def readDir(dir_content,list_items):#gives list of lists because each directory has n files and each file has m lines
    for file in dir_content:
        try:
            with open(file, encoding='utf8', errors='ignore') as f:
                for line in f:
                    text = [line.strip(string.punctuation) for line in line.lower().split()]  # remove punctuations from list
                    space_removed_list = list(filter(None, text))
                    list_items.append(space_removed_list)
        except IOError as e:
            print("ERROR:{}".format(e));
    return list_items, len(dir_content)


def convert_for_stemming(multi_list, spam_ham_list):  # converts entire vocabulary of spam/ham directory into a single list
    global stopwords_list
    for i in multi_list:
        for j in i:
            if len(stopwords_list) > 0:
                if j in stopwords_list:
                    continue
                else:
                    spam_ham_list.append(j)
            else:
                spam_ham_list.append(j)
    return spam_ham_list


def stem_list(list_to_stem,stemmed_list):  # convert entire list of lists into single 1D list
    for words in list_to_stem:
        stemmed_list.append(ps.stem(words))
    return stemmed_list


def populate_dictionary(list_item,dictionary_to_populate):  # populate spam ham dictionary with stemmed vocab
    for word in list_item:
            if dictionary_to_populate.get(word,0) == 0:
                dictionary_to_populate[word] = 1
            else:
                dictionary_to_populate[word] = dictionary_to_populate[word] + 1
    return dictionary_to_populate


def execute_train_commands(train_ham_list, train_spam_list):
    global total_no_of_docs_in_ham
    global total_no_of_docs_in_spam
    global train_ham_dictionary
    global train_spam_dictionary

    train_ham_list_new, total_no_of_docs_in_ham = readDir(train_ham_files, train_ham_list)
    train_spam_list_new, total_no_of_docs_in_spam = readDir(train_spam_files,train_spam_list)

    train_single_ham_list = []
    train_single_spam_list = []
    train_single_ham_list_new = convert_for_stemming(train_ham_list_new, train_single_ham_list)
    train_single_spam_list_new = convert_for_stemming(train_spam_list_new,train_single_spam_list)

    train_stem_ham_list = []
    train_stem_spam_list = []

    train_stem_ham_list_new = stem_list(train_single_ham_list_new, train_stem_ham_list)
    train_stem_spam_list_new = stem_list(train_single_spam_list_new,train_stem_spam_list)

    train_ham_dictionary = populate_dictionary(train_stem_ham_list_new, train_ham_dictionary)
    train_spam_dictionary = populate_dictionary(train_stem_spam_list_new, train_spam_dictionary)


def classify_data(dir_content,unique_set):
    labels = []
    for file in dir_content:
        current_ham_prob = 1
        current_spam_prob = 1
        with open(file) as f:
            for line in f:
                text = [line.strip(string.punctuation) for line in line.lower().split()]  # remove punctuations from list
                for word in text:
                    if train_ham_dictionary.get(word,0) + train_spam_dictionary.get(word,0) == 0:
                        continue

                    current_ham_prob *= float(total_no_of_docs_in_ham / (total_no_of_docs_in_ham + total_no_of_docs_in_spam)) * (float(train_ham_dictionary.get(word,0) + 1 )/(sum(train_ham_dictionary.values())+ unique_set))

                    current_spam_prob *= float(
                            total_no_of_docs_in_spam / (total_no_of_docs_in_ham + total_no_of_docs_in_spam)) * (
                                      float(train_spam_dictionary.get(word, 0) + 1) / float(
                                          sum(train_spam_dictionary.values()) + unique_set))

            if current_ham_prob >= current_spam_prob:
                labels.append(0)
            else:
                labels.append(1)
    return labels


def accuracy(spam, ham):
    total = len(spam) + len(ham)
    correct = sum(1 for x in spam if x == 1) + sum(1 for x in ham if x == 0)
    final_result = float(correct) / float(total)
    return 100* final_result


def unique_keys():
    ham_keys = set(train_ham_dictionary.keys())
    spam_keys = set(train_spam_dictionary.keys())
    intersection = ham_keys & spam_keys
    result = len(ham_keys)+len(spam_keys) - len(intersection)
    return result


def load_stopwords():
    try:
        with open(stopwords_file, 'r') as f:
            stopwords_list = f.read().split()
    except IOError as e:
        print("ERROR:{}".format(e))
    return stopwords_list


def main():
    getpaths()
    global train_ham_files
    global train_spam_files
    global test_ham_files
    global test_spam_files
    global stopwords_file
    global stopwords_list
    stopwords_list = load_stopwords()
    train_ham_list = []
    train_spam_list = []
    execute_train_commands(train_ham_list, train_spam_list)  # call training functions
    result = unique_keys()
    ham_label = classify_data(test_ham_files, result)
    spam_label = classify_data(test_spam_files, result)
    correct_prediction = accuracy(spam_label, ham_label)
    print(correct_prediction)

if __name__ == '__main__':
    main()
