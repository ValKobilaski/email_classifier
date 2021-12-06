import contractions
import email.parser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import re
import sys
import stat
import string
import numpy as np



def main():
    training_data = os.getcwd()+'/'+'CSDMC2010_SPAM/TRAINING/'
    processed_data_targ = 'CSDMC2010_SPAM/CLEAN' 
    clean_dir(training_data, processed_data_targ)
    features = extract_features_dir(processed_data_targ)
    labels = extract_labels('CSDMC2010_SPAM/SPAMTrain.label')
    feature_set = np.concatenate((features.T,labels[:, None].T), axis = 0)
    np.savetxt("features.csv", feature_set.T, delimiter=",", fmt='%f')


def clean_dir(dir_from, dir_to):
    """
    Helper function, to take in source directory and target
    directory and calls clean_file() on each file in the dir
    """

    # Error checking directories
    print('preprocessing data...')
    if not os.path.exists(dir_from):
        print('The source directory %s does not exist, exit...' % dir_from)
        sys.exit()
    if not os.path.exists(dir_to):
        print('The destination directory is newly created.')
        os.makedirs(dir_to)

    #Loop through files in dir
    files = os.listdir(dir_from)
    for file in files:
        src_file = os.path.join(dir_from, file)
        dest_file = os.path.join(dir_to, file)

        src_info = os.stat(src_file)
        #For subfolders, recurse
        if stat.S_ISDIR(src_info.st_mode):
            pass
        else: 
            #Clean file, and send to destinatoin folder
            output_text = clean_file(dir_from+file)
            with open(dir_to+'/'+file, 'w') as outfile:
                outfile.write(output_text)
    print('preprocessing complete')


def clean_file(file_name):
    """
    Takes in a file name, extracts body of email
    removes html present in .eml format as well as expands
    contractions for word matching. Retuns body as a str.
    """

    #Read file
    file = open(file_name, errors = 'ignore')
    body = email.message_from_file(file)
    file.close()

    #Get body of file
    body = body.get_payload()
    if type(body) == type(list()):
        body = body[0]
    if type(body) != type(''):
        body = str(body)

    #Clean file
    body = _remove_html_tags(body)
    body = _expand_contractions(body)
    return body


def extract_features_dir(src_dir):
    """
    Helper function, to extract features of every file
    within a source directory, returns numpy array of features
    """

    #Count files
    datapoints = len(os.listdir(src_dir))
    features = np.zeros((datapoints, 54))
    p = 0

    #Error checkign directory
    print('Extracting features...')
    if not os.path.exists(src_dir):
        print('The source directory %s does not exist, exit...' % src_dir)
        sys.exit()

    #Extract features of each file
    files = os.listdir(src_dir)
    for file in files:
        features[p] = extract_features_file(src_dir + '/' + file)
        p+= 1
    print('Feature Extraction complete')
    return features

    
def extract_features_file(file_name):
    """
    Takes in input file, reads file, extracts features
    and does further cleaning, returns numpy array
    of feature vector for specified file
    """

    #Read filter word file
    with open('filter_words.txt', 'r') as word_file:
        freq_words = word_file.read()
        freq_words = freq_words.split('\n')

    #Read input file
    with open(file_name, 'r') as infile:
        text = infile.read()
    text.replace("\n", " ")

    #Feature extraction and further cleaning
    char_data = _get_char_data(text)    #Returns features 0-2
    text = _remove_punctuation(text)
    capital_data = _get_capital_runs(text) #Returns features 3-5
    text = text.lower()
    word_data = _get_word_freq(freq_words,text)    #Returns features 6-54

    #combine features in to 1 vector
    data_vector = char_data + capital_data + word_data
    
    #Return Vector
    return np.array(data_vector)





def extract_labels(file_name):
    """
    Extracts output labels for specified file
    in our case "SPAMTrain.label"
    """
    y_labels = []
    # files = []

    #Read file line by line
    with open(file_name, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.split()
            y_labels.append(line[0])

    #Return numpy vector of labels
    return np.array(y_labels, dtype = float)


def _remove_html_tags(text):
    """
    Remove html tags from a string
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)


def _remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def _expand_contractions(text):
    return contractions.fix(text)


def _tokenize_string(text):
    return word_tokenize(text)


def _remove_stopwords(text_arr):
    filtered = []
    for word in text_arr:
        if word not in STOPWORDS:
            filtered.append(word)
    return filtered


def _stem_text(text_arr):
    stem_arr = []
    for word in text_arr:
        stem_arr.append(PS.stem(word))
    return stem_arr


def _get_char_data(text):
    """
    Reads string, returns chracter feature data,
    as standard python array
    """
    chars = {'!' : 0, '#' : 0, '$' : 0}
    for char in text:
        if char == '!':
            chars['!'] += 1
        elif char == '#':
            chars['#'] += 1
        elif char == '$':
            chars['$'] += 1
    return[chars['!'], chars['#'], chars['$']]


def _get_capital_runs(text):
    """
    Reads text, counts the number of capitals
    and the longest run of capitals >= 2.
    """
    capitals = []
    
    curr_count = 0
    curr_run = False

    for char in text:
        if char.isupper():
            curr_count +=1
        else:
            if (curr_count >= 2):
                #Keep track of run
                capitals.append(curr_count)
            curr_count = 0
    if(len(capitals) != 0 ):
        #In case no capitals avoid division by 0
        average = sum(capitals) / len(capitals)
        longest = max(capitals)
    else:
        average = 0
        longest = 0

    total = sum(capitals)

    return [average, longest, total]


def _get_word_freq(words, text):
    """
    Creates python dictionary from filter words
    reads file, and counts occurances of words
    """
    #Create dictionary
    count = {w : 0 for w in words}

    #Reads text and checks for filter words
    text_list = text.split(" ")
    for freq_word in count.keys():
        for word in text_list:
            if freq_word in word:
                count[freq_word] += 1
    
    #Creates data feature
    for freq_word in count.keys():
        count[freq_word] = count[freq_word] / len(text_list)

    #Returns python list for word frequencies
    return list(count.values())


if __name__ == "__main__":
    STOPWORDS = set(stopwords.words('english'))
    PS = PorterStemmer()
    
    main()