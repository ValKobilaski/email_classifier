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


def main():
    print('processing training data...')
    srcdir = os.getcwd()+'/'+'CSDMC2010_SPAM/TRAINING/'
    dstdir = 'processed_emails'
    if not os.path.exists(srcdir):
        print('The source directory %s does not exist, exit...' % srcdir)
        sys.exit()
    dstdir = 'processed_emails'
    if not os.path.exists(dstdir):
        print('The destination directory is newly created.')
        os.makedirs(dstdir)

    preprocess_dir(srcdir,dstdir)


def preprocess_dir(dir_from, dir_to):
    # Error checking directories
    if not os.path.exists(dir_from):
        print('The source directory %s does not exist, exit...' % dir_from)
        sys.exit()
    if not os.path.exists(dir_to):
        print('The destination directory is newly created.')
        os.makedirs(dstdir)

    files = os.listdir(dir_from)
    for file in files:
        src_file = os.path.join(dir_from, file)
        dest_file = os.path.join(dir_to, file)

        src_info = os.stat(src_file)
        if stat.S_ISDIR(src_info.st_mode):  # for subfolders, recurse
            #ExtractBodyFromDir(srcpath, dstpath)
            pass
        else: 
            output_text = preprocess_file(dir_from+file)
            with open(dir_to+'/'+file, 'w') as outfile:
                outfile.write(output_text)

def preprocess_file(file_name):

    #with open(file_name, errors='ignore') as file:
    file = open(file_name, errors='ignore')
    body = email.message_from_file(file)
    file.close()
    body = body.get_payload()
    #print(body)
    if type(body) == type(list()):
        body = body[0]  # only use the first part of payload
    if type(body) != type(''):
        body = str(body)

    body = body.lower()
    body = _remove_html_tags(body)
    body = _expand_contractions(body)
    body = _remove_punctuation(body)
    body_arr = _tokenize_string(body)
    body_arr = _stem_text(body_arr)
    return '\n'.join(body_arr)



        #print(body)
        #remove html tags
        #remove punctuation
        #removestop words
        #stem words


def _remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
            
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

if __name__ == "__main__":
    STOPWORDS = set(stopwords.words('english'))
    PS = PorterStemmer()
    
    main()