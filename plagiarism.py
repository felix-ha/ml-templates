import pandas as pd
import numpy as np
import os

import re
import operator


# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount

# Use function to label datatype for training 1 or test 2
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns=[datatype_var])

    # Prints counts by task and compare_dfcolumn for subset df
    # print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )

    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value

    # Performs stratified random sample of subset dataframe to create new df with subset values
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(
        lambda x: x.sample(min(len(x), sampling_number), random_state=sampling_seed))
    df_sampled = df_sampled.drop(columns=[datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value

    # Prints counts by compare_dfcolumn for selected sample
    # print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    # print("\nSampled DF:\n",df_sampled)

    # Labels all datatype_var column as train_value which will be overwritten to
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index:
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    # print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered


def train_test_dataframe(clean_df, random_seed=100):
    new_df = clean_df.copy()

    # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
    new_df.loc[:, 'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)

    # creating a dictionary of categorical:numerical mappings for plagiarsm categories
    mapping = {0: 'orig', 1: 'train', 2: 'test'}

    # traversing through dataframe and replacing categorical data
    new_df.Datatype = [mapping[item] for item in new_df.Datatype]

    return new_df


# helper function for pre-processing text given a file
def process_file(file):
    # put text in all lower case letters
    all_text = file.read().lower()

    # remove all non-alphanumeric chars
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)

    return all_text


def create_text_column(df, file_directory='datasets/plagiarism/data/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`.
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''

    # create copy to modify
    text_df = df.copy()

    # store processed text
    text = []

    # for each file (row) in the df, read in the file
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        # print(filename)
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)

    # add column to the copied dataframe
    text_df['Text'] = text

    return text_df


csv_file = 'datasets/plagiarism/data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
print(plagiarism_df.head())




print()


# Read in a csv file and return a transformed dataframe
def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things:
       1) converts `Category` column values to numerical values
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''

    plagiarism_df = pd.read_csv(csv_file)

    plagiarism_df['Class'] = np.select([plagiarism_df['Category'] == "non",
                                  plagiarism_df['Category'] == "heavy",
                                  plagiarism_df['Category'] == "light",
                                  plagiarism_df['Category'] == "cut",
                                  plagiarism_df['Category'] == "orig"],
                                 [0, 1, 1, 1, -1])

    plagiarism_df['Category'] = np.select([plagiarism_df['Category'] == "non",
                                     plagiarism_df['Category'] == "heavy",
                                     plagiarism_df['Category'] == "light",
                                     plagiarism_df['Category'] == "cut",
                                     plagiarism_df['Category'] == "orig"],
                                    [0, 1, 2, 3, -1])

    return plagiarism_df




# informal testing, print out the results of a called function
# create new `transformed_df`
transformed_df = numerical_dataframe(csv_file = csv_file)

# check work
# check that all categories of plagiarism have a class label = 1
print(transformed_df.head(10))

print()
# create a text column
text_df = create_text_column(transformed_df)
print(text_df.head())

row_idx = 0 # feel free to change this index

sample_text = text_df.iloc[0]['Text']

print('Sample processed text:\n\n', sample_text)

print()
random_seed = 1
complete_df = train_test_dataframe(text_df, random_seed=random_seed)
# check results
print(complete_df.head(10))

print()
print("ngrams")
print()

df = complete_df
n = 1
answer_filename = "g0pA_taska.txt"

from sklearn.feature_extraction.text import CountVectorizer


# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text,
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''

    # extracting text_A and text_S
    row = df[df['File'] == answer_filename]
    text_A = row['Text'].iloc[0]

    task = row['Task'].iloc[0]
    row_original = df[(df['Task'] == task) & (df['Datatype'] == 'orig')]
    text_S = row_original['Text'].iloc[0]

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(n, n))
    X = vectorizer.fit_transform([text_A, text_S])

    X_ndarray = X.toarray()
    n_gram_A = X_ndarray[0, :]

    sum_intersection = np.sum(np.min(X_ndarray, axis=0))
    sum_A = np.sum(n_gram_A)

    return sum_intersection / sum_A

print("")
print("lcs")
# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''

    answer_words = answer_text.split()
    source_words = source_text.split()

    answer_words_count = len(answer_words)
    source_words_count = len(source_words)

    matrix = np.zeros([source_words_count + 1, answer_words_count + 1], dtype=np.int)

    for row in np.arange(1, source_words_count + 1):
        for col in np.arange(1, answer_words_count + 1):

            if source_words[row-1] == answer_words[col-1]:
                matrix[row, col] = matrix[row-1, col-1] + 1
            else:
                matrix[row, col] = max(matrix[row-1, col], matrix[row, col-1])

    return matrix[-1, -1] / answer_words_count





# Run the test scenario from above
# does your function return the expected value?

A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"
S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"




# calculate LCS
lcs = lcs_norm_word(A, S)
print('LCS = ', lcs)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# Function returns a list of containment features, calculated for a given n
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    containment_values = []

    if (column_name == None):
        column_name = 'c_' + str(n)  # c_1, c_2, .. c_n

    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i, 'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks
        else:
            containment_values.append(-1)

    print(str(n) + '-gram containment features created!')
    return containment_values


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    lcs_values = []

    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i, 'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values


# Define an ngram range
ngram_range = range(1,3)


# The following code may take a minute to run, depending on your ngram_range
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i=0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)
    # create containment features
    all_features[i]=np.squeeze(create_containment_features(complete_df, n))
    i+=1

############# Calculate features for LCS_Norm Words
features_list.append('lcs_word')
all_features[i]= np.squeeze(create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# Print all features/columns
print()
print('Features: ', features_list)
print()


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)

# display shows all of a dataframe
print(corr_matrix)


# Takes in dataframes and a list of selected features (column names)
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''

    # get the training features
    train_x = None
    # And training class labels (0 or 1)
    train_y = None

    # get the test features and labels
    test_x = None
    test_y = None

    return (train_x, train_y), (test_x, test_y)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
test_selection = list(features_df)[:2] # first couple columns as a test
# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)