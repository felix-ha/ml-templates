from sklearn.datasets import load_files
reviews_train = load_files("../datasets/aclImdb")
# load_files returns a bunch, containing training texts and training labels

dataset = pickle.load( open( "save.p", "rb" ) )

text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[1]:\n{}".format(text_train[1]))