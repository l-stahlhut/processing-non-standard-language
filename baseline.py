# -*- coding: utf-8 -*-

# Processing non-standard language, HS21
# Author: Laura Stahlhut
# Date: 10.12.2021

"""This script performs automatic normalization on 'dev.txt' & 'test.txt' based on the most frequent normalization for
the tokens in 'train.txt'.
 - format of all 3 input files (one token per line): nonstandard\tstandard_gs\tPOS
 - format of the 2 output files: normalization_strategy\tnon_standard\tstandard_pred\tstandard_gs\tPOS_gs\t
"""

# Instructions on how to run the script:
# $ python3 baseline.py WUS_POS_data/train.txt WUS_POS_data/dev.txt WUS_POS_data/test.txt

import sys
from collections import Counter


# --------------------------------Step 1: create translation dictionary from train.txt -------------------------------

def get_tuples(tokens_train):
    """Input: Lines from a 3-column, tab seperated input file with one token per line (non-standard, standard, POS).
    Returns a list of tuples (non-standard, normalized_gold_standard).
    """
    word_pair_list = []

    for line in tokens_train:
        if line == '\n':
            word_pair_list.append('\n')  # preserve emtpy lines (sentence boundaries)
        else:
            word_pair = (line.split('\t')[0]), (line.split('\t')[1])  # non-standard, gold standard normalization
            word_pair_list.append(word_pair)

    return word_pair_list


def get_norm_freq(tokens_train):
    """Takes a list of tuples (word_pair_list) and returns a dictionary with the tuple as key and frequency of it
    occurring as value (word_pair_list_counts). E.g.:
    word_pair_list = [('I', 'ich'), ('hasses', 'hasse es'), ('hasses', 'hasse es'), ...].
    word_pair_list_counts = {('I', 'ich'): 1, ('hasses', 'hasse es'): 2, ...}.
    """
    word_pair_list = get_tuples(tokens_train)
    word_pair_list_counts = Counter(word_pair_list)

    return word_pair_list_counts


def translation_dict(tokens_train):
    """function that takes a dictionary in the form of 'freq_dict' and transforms it into the form of 'd',
     such that each key is unique and the value contains possible normalizations and their frequencies, e.g.
        freq_dict = {('I', 'ich'): 1, ('hasses', 'hasse es'): 2, ...}
        d = {'merci': [('merci', 5)],'viiu': [('viel', 3), ('viele', 1)], 'vill': [('viel', 15), ('viele', 6)], ...}
    """
    word_pair_counts = get_norm_freq(tokens_train)
    d = {}

    for item in word_pair_counts.keys():  # keys in input dict are word pairs (non-standard, normalization)
        try:
            d[item[0]] = []  # key for all non-standard words and an empty list as the corresponding value

        except IndexError:  # handle empty lines
            pass

    # we now have a dictionary d with non-standard words as keys and empty values
    # thus, we need to fill in the values now:

    for key in d:
        for item in word_pair_counts.keys():
            try:
                if key == item[0]:  # if a non-standard word in the new dictionary also appears in the old dictionary
                    # add its normalization and the frequency of that normalization as a list item in the values list
                    # of the new dictionary
                    d[key].append((item[1], word_pair_counts[item]))
            except IndexError:  # empty lines
                pass
    return d


# -----------------Step 2: perform automatic normalization on 'test.txt' and 'dev.txt' ------------------------------

def normalize(tokens_train, tokens_dev_or_test, path_in):
    d = translation_dict(tokens_train)

    path_out = path_in + "_norm_out.txt"  # e.g. 'WUS_POS_data/dev_norm_out.txt'

    with open(path_out, 'w', encoding='utf8') as f_out:

        for line in tokens_dev_or_test:

            if line == "\n":  # preserve empty lines (sentence boundaries)
                f_out.write("\n")
            else:
                non_standard = line.split('\t')[0]
                standard = line.split('\t')[1]
                POS = line.split('\t')[2]

                if non_standard in d:  # if word in first column exists as dictionary key, look for normalization
                    list_tuples = d[non_standard]  # values are list of tuples

                    # UNIQUE: only one normalization per non-standard word --> take that one as normalization
                    if len(list_tuples) == 1:  #
                        n_strategy = "U"
                        n_pred = list_tuples[0][0]  # the value of the inner dictionary = normalization
                        line_out = n_strategy + "\t" + non_standard + "\t" + n_pred + "\t" + standard + "\t" + POS
                        f_out.write(line_out)

                    # AMBIGUOUS: multiple normalizations per non-standard word -> pick most frequent one
                    else:
                        n_strategy = "A"
                        n_pred = max(list_tuples, key=lambda x: x[1])[0]
                        line_out = n_strategy + "\t" + non_standard + "\t" + n_pred + "\t" + standard + "\t" + POS
                        f_out.write(line_out)

                # NEW: new words (not in translation dictionary) --> take non-standard word as normalisation
                else:
                    n_strategy = "N"
                    n_pred = non_standard
                    line_out = n_strategy + "\t" + non_standard + "\t" + n_pred + "\t" + standard + "\t" + POS
                    f_out.write(line_out)

    return f_out


def main():
    # file to get translation dictionary (train.txt)
    with open(sys.argv[1], 'r', encoding='utf8') as f_train:
        train_lines = f_train.readlines()

    # files to perform automatic normalization on (dev.txt, test.txt)
    with open(sys.argv[2], 'r', encoding='utf8') as f_dev:
        dev_lines = f_dev.readlines()
        dev_path_in = f_dev.name.rstrip(".txt")  # get path in order to create name of output file
    
    with open(sys.argv[3], 'r', encoding='utf8') as f_test:
        test_lines = f_test.readlines()
        test_path_in = f_test.name.rstrip("txt").rstrip(".")  # get path in order to create name of output file

    normalize(train_lines, dev_lines, dev_path_in)
    normalize(train_lines, test_lines, test_path_in)


if __name__ == "__main__":
    main()
