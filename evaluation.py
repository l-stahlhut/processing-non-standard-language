# -*- coding: utf-8 -*-

# Processing non-standard language, HS21
# Author: Laura Stahlhut
# Date: 10.12.2021

""" This script evaluates a pre-trained POS-tagger from spaCy in different normalization settings and creates
an evaluation report.
- upper bound: evaluate tagger on manually normalized version.
- lower bound: evaluate tagger on original text (non-standard).
- baseline: evaluate tagger on our automatically normalized text (from baseline.py).

- format of the input text files:
    norm_strategy\tnon_standard\tstandard_pred\tstandard_gs\tPOS_gs
- format of the output text files:
    norm_strategy\tnon_standard\tstandard_pred\tstandard_gs\tPOS_gs\tPOS_upper_bound\tPOS_lower_bound\tPOS_baseline
"""

# Instructions on how to run the script:
# 1) Requirements: install spacy for German and tabulate in a virtual environment
# $ python3 -m venv myenv
# $ source myenv/bin/activate
# $ pip install -U pip setuptools wheel
# $ pip install -U spacy
# $ python -m spacy download de_core_news_sm
# $ pip install tabulate

# 2) Run the script like so: #todo
# $ python3 evaluation.py WUS_POS_data/dev_norm_out.txt WUS_POS_data/test_norm_out.txt

import spacy
from tabulate import tabulate
import sys

nlp = spacy.load('de_core_news_sm')  # load model

# ---------------------------------- Pre-processing a text for POS tagging -----------------------------------------
# SpaCy expects sentences as inputs, not tokens or an entire text.
# get lists of sentences for dev & text data for 3 settings each (lower bound, upper bound, baseline)


def columns_to_lists(lines):
    """
    Takes a tokenized text as input where the entire text is a list (one token per list entry) and each token has the
    following information (tab seperated): Normalization strategy, non-standard, automatic normalization,
    normalization gold standard, POS gold standard:
    Input = ['A\ti\tich\tich\tPPER\n', 'U\tmuen\tmuss\tmuss\tVMFIN\n', 'U\talles\talles\talles\tPIS\n', ...]
    Returns the entire tokenized text in three lists: lower bound, upper bound, baseline
    Output = (['i', 'muen', 'alles', 'wüsse', ...],['ich', 'muss', 'alles', ...],['ich', 'muss', 'alles', ...])
    """
    non_standard_list = []  # 1) lower bound -> column [1] = non_standard
    norm_gs_list = []       # 2) upper bound -> column [3] = gold standard normalization
    norm_aut_list = []      # 3) baseline -> column [2] = automatic normalization

    # fill lists with data for upper bound, lower bound, baseline
    for line in lines:
        if line == "\n":
            non_standard_list.append('\n')  # empty lines (sentence boundaries)
            norm_gs_list.append('\n')
            norm_aut_list.append('\n')
        else:
            non_standard_list.append(line.split('\t')[1].rstrip())   # lower bound data
            norm_gs_list.append(line.split('\t')[3].rstrip())        # upper bound data
            norm_aut_list.append(line.split('\t')[2].rstrip())       # baseline data

    return non_standard_list, norm_gs_list, norm_aut_list


def sent_boundaries(lines):
    """Helper function to create sentences from a tokenized text. This funciton returns indexes for sentence boundaries.
    Input = ['A\ti\tich\tich\tPPER\n', '\n', 'U\talles\talles\talles\tPIS\n', ...]
    Output = [1, 5, 12, 15, 17, 23, 33, 47, ...]
    """
    lines.append('\n')  # add newline at the end (needed for extract_sent function)
    sent_boundaries_indexes = []

    for i in range(len(lines)):
        if lines[i] == '\n':
            sent_boundaries_indexes.append(i)

    return sent_boundaries_indexes


def extract_sentences(text, lines):
    """Takes a tokenized text as input and returns sentences.
    Input:
    - text = ['i', 'muen', 'alles', 'wüsse', ...],
    - lines = original file we're working with (either dev_lines or text_lines)
    Output = ['i muen alles wüsse XD', 'Nei nur umeglege bis am 8', 'Die wahrheit']
    """
    text.append('\n')  # needed in the for loop, so the last sentence will be added to sentences list

    sentence = []
    sentences = []
    for i, token in enumerate(text):
        if i in sent_boundaries(lines):  # locate sentence boundary
            sentence = ' '.join(sentence)
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(token)

    return sentences


# -------------------------------------------- POS tagging -----------------------------------------
# tag all 6 lists (dev & text data for 3 settings each (lower bound, upper bound, baseline))


def predict_pos(text):
    """Takes a list of sentences and returns a list of tuples with token and pos tag.
    Output = [('und', 'CCONJ'), ('extrem', 'ADJ'), ('vill', 'PROPN'), ...]
    This tokenizes the text differently from the input file. Output needs to be postprocessed in order to match up with
    lines in dev- and text files.
    """

    POS_predicted = []

    for sent in text:
        doc = nlp(sent)

        for token in doc:
            POS_predicted.append((token.text, token.tag_))

        POS_predicted.append('\n')

    return POS_predicted


def correct_pos(text, orig_tokens):
    """Function to correct the output by predict_pos() which tokenizes the text differently form the tokenized input
    text. This function firstly merges tokens that were tokenized by spacy even though they shouldn't be (e.g. '!!!')
    and then iterates through the SpaCy output and the original text simultaneously in order to match up many-to-one
    cases.
        Input:
        - original tokenized text = ['hasse es', '\n', 'naja', 'random', 'würde ich', 'nicht', 'sagen', ...]
        - predict_pos(text, orig_tokens) = [('und', 'CCONJ'), ('extrem', 'ADJ'), ('vill', 'PROPN'), ...]
    Note: In order to check if the tokenization is correct, it is possible to change the function to return the variable
    correct_pos instead and print it. This way, POS and respective word get printed side-by side.
        - correct_pos  = [('hasse', 'NN'), ('es', 'PPER'), '\n', ('naja', 'NE'), ...]
        Output: POS_only = ['ADV', 'NN', 'ADJA', 'NN', '$,', ...]
    """
    # predict POS for a specified text column
    predicted_POS = predict_pos(text)

    # dictionary for cases that are not tokenized in the input file but get tokenized by SpaCY
    # the cases were manually discovered by comparing input column to the corresponding SpaCY output
    d = {
        '!!!': ('$.', 3),
        '!!': ('$.', 2),
        '!!!!!': ('$.', 5),
        '??': ('$.', 2),
        '?!': ('$.', 2),
        '*_*': ('EMO', 3),
        ';P': ('EMO', 2),
        ':\\': ('$.', 2),
        'evt.': ('ADV', 2),
        '[StreetAddress]': ('NN', 3),
        '[LastName]': ('NN', 3),
        '*google*': ('NE', 3),
        'Z.b.': ('NN', 2)
    }

    correct_POS = []  # list of tuples with (correct word, correct POS)
    # note: the word is not necessary for the final output but it was needed in order to be able to compare input
    # columns with spacy columns.

    i = 0  # counter for original tokenized text (list of tokens)

    for word in orig_tokens:

        # preserve empty lines
        if len(word.split()) == 0:
            correct_POS.append("\n")
            i += 1

        # one-token expression --> fill tuple with word and corresponding tag
        elif len(word.split()) == 1:  # one-token word --> fill tuple with word and corresponding tag

            # handle cases spacy tokenizes that aren't tokenized in the input file
            if word in d.keys():
                correct_POS.append((word, d[word][0]))
                i += d[word][1]  # tokenized in spacy
            else:
                correct_POS.append((predicted_POS[i][0], predicted_POS[i][1]))
                i += 1

        # multi-token word --> connect tags with '+'
        elif len(word.split()) == 2:
            word_1, tag_1 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_2, tag_2 = predicted_POS[i][0], predicted_POS[i][1]
            correct_POS.append((word_1 + '+' + word_2, tag_1 + '+' + tag_2))
            i += 1

        elif len(word.split()) == 3:
            word_1, tag_1 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_2, tag_2 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_3, tag_3 = predicted_POS[i][0], predicted_POS[i][1]
            correct_POS.append((word_1 + '+' + word_2 + '+' + word_3, tag_1 + '+' + tag_2 + '+' + tag_3))
            i += 1

        elif len(word.split()) == 4:  # maximum token length in this setting is 4
            word_1, tag_1 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_2, tag_2 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_3, tag_3 = predicted_POS[i][0], predicted_POS[i][1]
            i += 1
            word_4, tag_4 = predicted_POS[i][0], predicted_POS[i][1]
            correct_POS.append(
                (word_1 + '+' + word_2 + '+' + word_3 + '+' + word_4, tag_1 + '+' + tag_2 + '+' + tag_3 + '+' + tag_4))
            i += 1

    # corract_POS = a list of (word, POS) tuples --> for checking where the text is tokenized differently by spaCy
    correct_POS = correct_POS[:-1]  # remove the last newline that was added in the beginning of the sentence function

    # we only need POS in the output, not the tuples
    POS_only = []

    i = 0

    for item in correct_POS:
        if item == '\n':
            POS_only.append('\n')
            i += 1
        else:
            POS_only.append(item[1])  # append only the tag, word is not needed
            i += 1

    return POS_only

# -------------------------------------------- Write 8 column files  -----------------------------------------


def write_outfile(orig_lines, POS_lb, POS_ub, POS_bl, path_in):
    """Writes a new 8 column file with the following informaiton:
    Norm. Strategy | non-standard | normaliz. predicted | normaliz. gs | POS gs | POS lb | POS ub | POS baseline
    :param:
        - orig_lines = lines in original input file
        - POS_lb = Lower bound setting --> POS of original non-standard text
        - POS_ub = Upper bound setting --> POS of gold standard normalization (manual)
        - POS_bl = Baseline setting --> POS of automatically normalized text (performed by baseline.py)
    """

    path_out = path_in + "POS_out.txt"  # e.g. 'WUS_POS_data/dev_norm_POS_out.txt'

    with open(path_out, 'w', encoding='utf8') as f_out:
        i = 0
        for line in orig_lines:

            if line == "\n":  # keep empty lines (sentence boundaries)
                f_out.write("\n")
                i += 1
            else:
                f_out.write(line.rstrip() + '\t' + POS_lb[i] + '\t' + POS_ub[i] + '\t' + POS_bl[i] + '\n')
                i += 1

    return path_out

# ---------------------------------- Accuracy calculation -----------------------------------------


def calculate_accuracies(infile_path):

    # set up counters:
    # 1) total number per normalization strategy
    n_unique, n_ambigue, n_new = 0, 0, 0

    # 2) counter for how often predicted POS = gold standard POS
    same_u_lb, same_a_lb, same_n_lb = 0, 0, 0   # lower bound: unique, ambigue, new
    same_u_ub, same_a_ub, same_n_ub = 0, 0, 0   # upper bound: unique, ambigue, new
    same_u_b, same_a_b, same_n_b = 0, 0, 0      # baseline: unique, ambigue, new

    # ----------- get counts needed for accurady calculation -----------

    with open(infile_path, 'r', encoding='utf8') as infile:
        lines = infile.readlines()

        for line in lines:
            # UNIQUE
            if line.startswith('U'):
                n_unique += 1
                if line.split('\t')[4] == line.split('\t')[5]:           # lower bound
                    same_u_lb += 1
                if line.split('\t')[4] == line.split('\t')[6]:           # upper bound
                    same_u_ub += 1
                if line.split('\t')[4] == line.split('\t')[7].rstrip():  # baseline
                    same_u_b += 1
            # AMBIGUE
            elif line.startswith('A'):
                n_ambigue += 1
                if line.split('\t')[4] == line.split('\t')[5]:           # lower bound
                    same_a_lb += 1
                if line.split('\t')[4] == line.split('\t')[6]:           # upper bound
                    same_a_ub += 1
                if line.split('\t')[4] == line.split('\t')[7].rstrip():  # baseline
                    same_a_b += 1
            # NEW
            elif line.startswith('N'):
                n_new += 1
                if line.split('\t')[4] == line.split('\t')[5]:           # lower bound
                    same_n_lb += 1
                if line.split('\t')[4] == line.split('\t')[6]:           # ub
                    same_n_ub += 1
                if line.split('\t')[4] == line.split('\t')[7].rstrip():  # baseline
                    same_n_b += 1
            # newlines
            else:
                pass

    n_total = n_unique + n_ambigue + n_new

    # -------- calculate accuracies -----------
    # lower bound
    accuracy_lb_u = round((same_u_lb / n_unique)*100, 2)     # unique
    accuracy_lb_a = round((same_a_lb / n_ambigue)*100, 2)     # ambigue
    accuracy_lb_n = round((same_n_lb / n_new)*100, 2)         # new

    accuracy_total_lb = round(((same_u_lb + same_a_lb + same_n_lb) / n_total)*100, 2)

    # Upper bound
    accuracy_ub_u = round((same_u_ub / n_unique)*100, 2)      # unique
    accuracy_ub_a = round((same_a_ub / n_ambigue)*100, 2)     # ambigue
    accuracy_ub_n = round((same_n_ub / n_new)*100, 2)         # new

    accuracy_total_ub = round(((same_u_ub + same_a_ub + same_n_ub) / n_total)*100, 2)

    # Baseline
    accuracy_b_u = round((same_u_b / n_unique)*100, 2)    # unique
    accuracy_b_a = round((same_a_b / n_ambigue)*100, 2)   # ambigue
    accuracy_b_n = round((same_n_b / n_new)*100, 2)       # new

    accuracy_total_b = round(((same_u_b + same_a_b + same_n_b) / n_total)*100, 2)

    return (n_unique, n_ambigue, n_new, n_total), \
           (accuracy_lb_u, accuracy_lb_a, accuracy_lb_n, accuracy_total_lb), \
           (accuracy_ub_u, accuracy_ub_a, accuracy_ub_n, accuracy_total_ub), \
           (accuracy_b_u, accuracy_b_a, accuracy_b_n, accuracy_total_b)

# ------------------------------------- Evaluation report -----------------------------------------


def write_report(dev_accuracies, test_accuracies):

    # lower bound data
    table_lb = [['Case', 'N (Dev)', 'Accuracy (Dev)', 'N (Test)', 'Accuracy (Test)'],
                ['Unique', dev_accuracies[0][0], dev_accuracies[1][0], test_accuracies[0][0], test_accuracies[1][0]],
                ['Ambiguous', dev_accuracies[0][1], dev_accuracies[1][1], test_accuracies[0][1], test_accuracies[1][1]],
                ['New', dev_accuracies[0][2], dev_accuracies[1][2], test_accuracies[0][2], test_accuracies[1][2]],
                ['Total', dev_accuracies[0][3], dev_accuracies[1][3], test_accuracies[0][3], test_accuracies[1][3]]]

    # upper bound data
    table_ub = [['Case', 'N (Dev)', 'Accuracy (Dev)', 'N (Test)', 'Accuracy (Test)'],
                ['Unique', dev_accuracies[0][0], dev_accuracies[2][0], test_accuracies[0][0], test_accuracies[2][0]],
                ['Ambiguous', dev_accuracies[0][1], dev_accuracies[2][1], test_accuracies[0][1], test_accuracies[2][1]],
                ['New', dev_accuracies[0][2], dev_accuracies[2][2], test_accuracies[0][2], test_accuracies[2][2]],
                ['Total', dev_accuracies[0][3], dev_accuracies[2][3], test_accuracies[0][3], test_accuracies[2][3]]]

    # baseline data
    table_b = [['Case', 'N (Dev)', 'Accuracy (Dev)', 'N (Test)', 'Accuracy (Test)'],
                ['Unique', dev_accuracies[0][0], dev_accuracies[3][0], test_accuracies[0][0], test_accuracies[3][0]],
                ['Ambiguous', dev_accuracies[0][1], dev_accuracies[3][1], test_accuracies[0][1], test_accuracies[3][1]],
                ['New', dev_accuracies[0][2], dev_accuracies[3][2], test_accuracies[0][2], test_accuracies[3][2]],
                ['Total', dev_accuracies[0][3], dev_accuracies[3][3], test_accuracies[0][3], test_accuracies[3][3]]]

    # write tables to file
    with open("WUS_POS_data/eval_report.txt", 'w', encoding='utf8') as eval_report:
        eval_report.write("Lower bound:\n\n")
        eval_report.write(tabulate(table_lb, headers='firstrow'))
        eval_report.write("\n\nUpper bound:\n\n")
        eval_report.write(tabulate(table_ub, headers='firstrow'))
        eval_report.write("\n\nBaseline:\n\n")
        eval_report.write(tabulate(table_b, headers='firstrow'))

    return None


def main():
    # Open input files (=output files of baseline.py)
    with open(sys.argv[1], 'r', encoding='utf8') as f_dev_in:
        dev_lines = f_dev_in.readlines()
        dev_path_in = f_dev_in.name.rstrip("out.txt")  # get path in order to create name of output file

    with open(sys.argv[2], 'r', encoding='utf8') as f_test_in:
        test_lines = f_test_in.readlines()
        test_path_in = f_test_in.name.rstrip("out.txt")  # get path in order to create name of output file

    # ---------------------------------- process DEVELOPMENT DATA -----------------------------------
    # get 3 lists of tokens for our 3 settings (lower bound, upper bound, baseline)
    non_standard_dev, norm_gs_dev, norm_aut_dev = columns_to_lists(dev_lines)

    # extract sentences
    # for some reason, extract_sentences function updates dev_lines, thus we make a copy of it
    dev_lines2 = dev_lines.copy()
    non_standard_dev_sent = extract_sentences(non_standard_dev, dev_lines2)
    norm_gs_dev_sent = extract_sentences(norm_gs_dev, dev_lines2)
    norm_aut_dev_sent = extract_sentences(norm_aut_dev, dev_lines2)

    # POS tagging
    non_standard_dev_pos = correct_pos(non_standard_dev_sent, non_standard_dev)  # lower bound POS
    norm_gs_dev_pos = correct_pos(norm_gs_dev_sent, norm_gs_dev)                 # upper bound POS
    norm_aut_dev_pos = correct_pos(norm_aut_dev_sent, norm_aut_dev)              # baseline POS

    # write 8-column-file with new POS tags
    write_outfile(dev_lines, non_standard_dev_pos, norm_gs_dev_pos, norm_aut_dev_pos, dev_path_in)
    new_dev_path = write_outfile(dev_lines, non_standard_dev_pos, norm_gs_dev_pos, norm_aut_dev_pos, dev_path_in)

    # ------------------------------------ process TEST DATA -------------------------------------
    # get 3 lists of tokens for our 3 settings (lower bound, upper bound, baseline)
    non_standard_test, norm_gs_test, norm_aut_test = columns_to_lists(test_lines)

    # extract sentences
    test_lines2 = test_lines.copy()
    non_standard_test_sent = extract_sentences(non_standard_test, test_lines2)
    norm_gs_test_sent = extract_sentences(norm_gs_test, test_lines2)
    norm_aut_test_sent = extract_sentences(norm_aut_test, test_lines2)

    # POS tagging
    non_standard_test_pos = correct_pos(non_standard_test_sent, non_standard_test)  # lower bound POS
    norm_gs_test_pos = correct_pos(norm_gs_test_sent, norm_gs_test)                 # upper bound POS
    norm_aut_test_pos = correct_pos(norm_aut_test_sent, norm_aut_test)              # baseline POS

    # write 8-column-file with new POS tags
    write_outfile(test_lines, non_standard_test_pos, norm_gs_test_pos, norm_aut_test_pos, test_path_in)
    new_test_path = write_outfile(test_lines, non_standard_test_pos, norm_gs_test_pos, norm_aut_test_pos, test_path_in)

    # ----------------------------------- accuracy report ---------------------------------------
    # calculate accuracies
    dev_accuracies = calculate_accuracies(new_dev_path)
    test_accuracies = calculate_accuracies(new_test_path)

    # write report
    write_report(dev_accuracies, test_accuracies)


if __name__ == '__main__':
    main()
