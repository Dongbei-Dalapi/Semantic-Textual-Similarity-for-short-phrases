from deeplearning_model import SiameseLSTM
from sklearn.preprocessing import StandardScaler
from data_process import load_data, get_char_id
from sklearn import svm
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from traditional_nlp_model import extract_features
import warnings
warnings.filterwarnings("ignore")


def is_similar(p):
    if p == 1:
        return "similar"
    else:
        return "not similar"


if __name__ == '__main__':
    # set up the argument
    argp = argparse.ArgumentParser(description='Build a classifier.')
    argp.add_argument('--epoch', default=50, type=int,
                      help='number of training epochs (default: 100)')
    argp.add_argument('--batch_size', default=64, type=int,
                      help='batch Size (default: 64)')
    argp.add_argument('--train', action='store_true',
                      help='train the model or not. If not, do evaluaion only (default: False).')
    argp.add_argument('--save', default='SiameseLSTM.h5',
                      help='specify the model name to save (default: SiameseLSTM.h5)')
    args = argp.parse_args()

    # load data
    train_text, test_text, train_labels, test_labels = load_data()
    ori_test_text = []
    with open('./data/test_data.txt') as f:
        lines = f.readlines()
        for line in lines:
            t = line.strip().split(',')
            ori_test_text.append((t[0].strip(), t[1].strip()))

    # get character based sequence
    train_char1, train_char2, char2ind = get_char_id(train_text)
    test_char1, test_char2, _ = get_char_id(test_text)
    ori_test_char1, ori_test_char2, _ = get_char_id(ori_test_text)
    vocab_size = len(char2ind)

    # build model
    lstm = SiameseLSTM(vocab_size, epoch=args.epoch,
                       batch_size=args.batch_size)
    if args.train:
        lstm.train(train_char1, train_char2, train_labels)
        lstm.save(args.save)
    else:
        lstm.load(args.save)

    p = lstm.predict(test_char1, test_char2)

    # Evaluation
    precision, recall, f1score, _ = precision_recall_fscore_support(
        test_labels, p, average='macro')
    print('Accuracy: %.3f' % accuracy_score(test_labels, p))
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    print('F1 Score: %.3f' % f1score)
    print('--------------------------')
    print('{:<30s}{:<30s}{:<20s}'.format(
        'Job title 1', 'Job title 2', 'Similarity'))
    print('-----------------------------------------------------------------------')
    for text, c1, c2 in zip(ori_test_text, ori_test_char1, ori_test_char2):
        print('{:<30s}{:<30s}{:<20s}'.format(
            text[0], text[1], is_similar(lstm.predict(c1.reshape(1, -1), c2.reshape(1, -1)))))
    print('-----------------------------------------------------------------------')
