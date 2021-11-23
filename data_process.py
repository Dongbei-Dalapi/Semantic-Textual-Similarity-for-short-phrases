from os import access
from imblearn import under_sampling
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize


def load_data(filename='./data/train_data.csv'):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        jobs = np.array(list(reader))
    titles = []
    labels = []
    row = jobs.shape[0]
    column = jobs.shape[1]
    for i in range(row):
        for j in range(column):
            job1 = jobs[i][j]
            if job1 == '':
                continue
            for same_c in range(j+1, column):
                job2 = jobs[i][same_c]
                if job2 == '':
                    break
                titles.append((job1.lower().strip(), job2.lower().strip()))
                labels.append(1)
            for notsimilar_r in range(i+1, row):
                for notsimilar_c in range(column):
                    job2 = jobs[notsimilar_r][notsimilar_c]
                    if job2 == '':
                        break
                    titles.append((job1.lower().strip(), job2.lower().strip()))
                    labels.append(0)
    # deal with imbalance data
    oversample = RandomOverSampler()
    titles, labels = oversample.fit_resample(titles, labels)
    return train_test_split(titles, labels, test_size=0.3)


def get_char_id(phrases):
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
    chars = []
    for c in CHAR_DICT:
        chars.append(c)
    char_indices = dict((c, i+1) for i, c in enumerate(chars))
    en_char1 = []
    en_char2 = []
    for phrase in phrases:
        text1 = [word.lower() for word in word_tokenize(phrase[0])]
        text2 = [word.lower() for word in word_tokenize(phrase[1])]
        encoded_char_text1 = []
        encoded_char_text2 = []
        for word in text1:
            char_level = []
            for char in word:
                if char in char_indices:
                    char_level.append(char_indices[char])
                else:
                    char_level.append(0)
            encoded_char_text1.append(char_level)
        for word in text2:
            char_level = []
            for char in word:
                if char in char_indices:
                    char_level.append(char_indices[char])
                else:
                    char_level.append(0)
            encoded_char_text2.append(char_level)
        en_char1.append(pad_sequences(encoded_char_text1,
                        20, padding='post').flatten())
        en_char2.append(pad_sequences(encoded_char_text2,
                        20, padding='post').flatten())
    en_char1 = pad_sequences(en_char1, 100,  padding='post')
    en_char2 = pad_sequences(en_char2, 100,  padding='post')
    return en_char1, en_char2, char_indices
