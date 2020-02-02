import os
import glob
import pickle
import numpy as np


def read_imdb_data(data_dir='../datasets/aclImdb'):
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)



            for f in files:
                with open(f) as review:

                    try:
                        data[data_type][sentiment].append(review.read())
                        # Here we represent a positive review by '1' and a negative review by '0'
                        labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    except Exception as e:
                        print(e)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels

def save_imdb_set():
    data, labels = read_imdb_data()
    print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
                len(data['train']['pos']), len(data['train']['neg']),
                len(data['test']['pos']), len(data['test']['neg'])))

    dataset = { "data": data, "labels": labels }

    pickle.dump(dataset, open( "save.p", "wb" ) )

def get_imdb_data():


    dataset = pickle.load( open( "save.p", "rb" ) )


    # loading train set
    #text_train, y_train = dataset['data']['train'], labels = dataset['labels']
    text_train = dataset['data']['train']['pos'] + dataset['data']['train']['neg']
    y_train = dataset['labels']['train']['pos'] + dataset['labels']['train']['neg']


    print("type of text_train: {}".format(type(text_train)))
    print("length of text_train: {}".format(len(text_train)))
    print("text_train[1]:\n{}".format(text_train[1]))

    text_train = [doc.replace("<br />", " ") for doc in text_train]

    print("Samples per class (training): {}".format(np.bincount(y_train)))


    #loading test set
    text_test = dataset['data']['test']['pos'] + dataset['data']['test']['neg']
    y_test = dataset['labels']['test']['pos'] + dataset['labels']['test']['neg']


    print("type of text_test: {}".format(type(text_test)))
    print("length of text_test: {}".format(len(text_test)))
    print("text_train[1]:\n{}".format(text_test[1]))

    text_test = [doc.replace("<br />", " ") for doc in text_test]

    print("Samples per class (traning): {}".format(np.bincount(y_test)))

    return text_train, text_test, y_train, y_test

def train_simple_log_reg(text_train, y_train, N, max_features):
    from sklearn.utils import shuffle
    text_train, y_train = shuffle(text_train, y_train)



    text_train = text_train[0:N]
    y_train = y_train[0:N]

    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(max_features=max_features, stop_words='english')  # min_df=10)
    vect.fit(text_train)

    print("Vocabulary size: {}".format(len(vect.vocabulary_)))
    print("Vocabulary content:\n {}".format(vect.vocabulary_))

    bag_of_words = vect.transform(text_train)
    print("bag_of_words: {}".format(repr(bag_of_words)))
    print("Dense representation of bag_of_words:\n{}".format(
        bag_of_words.toarray()))

    feature_names = vect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("Features:\n{}".format(feature_names))

    X_train = vect.transform(text_train)

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    max_magnitude_coef = np.max(clf.coef_[0])

    print("max coefs:")
    print()

    for i in range(len(clf.coef_[0])):

        if np.absolute(clf.coef_[0][i]) > max_magnitude_coef / 2:
            print(round(clf.coef_[0][i], 2), " ", feature_names[i])

text_train, text_test, y_train, y_test = get_imdb_data()

train_simple_log_reg(text_train, y_train, N=24500, max_features=500)
