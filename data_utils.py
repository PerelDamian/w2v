import string
import re
import numpy as np

ascii_and_alphanumeric_chars = list(filter(str.isalnum, string.printable))


def get_word_pairs(s, inp_ind, win_size):
    min_cntxt_ind = max(0, inp_ind - win_size)
    max_cntxt_ind = min(len(s), inp_ind + win_size + 1)

    pairs = [(s[inp_ind], s[cntxt_ind]) for cntxt_ind in range(min_cntxt_ind, max_cntxt_ind) if cntxt_ind != inp_ind]

    return pairs


def get_sentence_pairs(s, win_size):
    pairs = []

    for inp_ind in range(len(s)):
        pairs += get_word_pairs(s, inp_ind, win_size)

    return pairs


def get_data_pairs(lst_of_sntnces, win_size, max_pairs=np.inf):
    pairs = []

    for s in lst_of_sntnces:
        pairs += get_sentence_pairs(s, win_size)
        if len(pairs) > max_pairs:
            break

    return pairs


def clean_sentence(sentence):
    # lower
    sentence = sentence.lower()

    # remove punctutation. it's better to replace with spaces and then convert multiple spaces to one.
    for ch in string.punctuation:
        sentence = sentence.replace(ch, ' ')
    sentence = re.sub('\s+', ' ', sentence).strip()

    # keep only alphanumeric chars
    sentence = ''.join(letter for letter in sentence if letter in ascii_and_alphanumeric_chars or letter == ' ')

    # delete word with less than 3 chars
    sentence = [word for word in sentence.split(' ') if len(word) >= 3]

    return sentence


def get_voc_maps(data):
    voc_list = ['<unk>'] + list(set([word for sntnce in data for word in sntnce]))
    voc_indexes = range(len(voc_list))

    return dict(zip(voc_list, voc_indexes)), dict(zip(voc_indexes, voc_list))


def convert_data_to_index_form(data, word2ind):
    data_index_form = []

    for s in data:
        s_ind_form = []
        for w in s:
            try:
                ind = word2ind[w]
            except KeyError:
                ind = word2ind['<unk>']
            s_ind_form.append(ind)
        if len(s_ind_form) > 1:
            data_index_form.append(s_ind_form)

    return data_index_form


class SentenceSplitter:
    def __init__(self, dataset_split_path):
        with open(dataset_split_path) as f:
            lines = f.readlines()
        lines = [[int(y) for y in x.strip().split(',')] for x in lines[1:]]

        self.train_indexes = [tup[0] for tup in lines if tup[1] == 1]
        self.test_indexes = [tup[0] for tup in lines if tup[1] == 2]
        self.val_indexes = [tup[0] for tup in lines if tup[1] == 3]

    def get_set_from_sentence_index(self, sentence_ind):
        if sentence_ind in self.train_indexes:
            return 1
        elif sentence_ind in self.test_indexes:
            return 2
        elif sentence_ind in self.val_indexes:
            return 3
        else:
            raise KeyError


class DataParser:
    def __init__(self, sentences_path, splitter):
        with open(sentences_path) as f:
            lines = f.readlines()
        lines = [x.strip().split('\t') for x in lines[1:]]

        sentences_indexes = [int(tup[0]) for tup in lines]
        cleaned_sentences = [clean_sentence(tup[1]) for tup in lines]

        self.train = []
        self.test = []
        self.val = []

        for ind, sntnce in zip(sentences_indexes, cleaned_sentences):
            set_ind = splitter.get_set_from_sentence_index(ind)
            if set_ind == 1:
                self.train.append(sntnce)
            elif set_ind == 2:
                self.test.append(sntnce)
            elif set_ind == 3:
                self.val.append(sntnce)

        self.word2ind, self.ind2word = get_voc_maps(self.train)

        self.train = convert_data_to_index_form(self.train, self.word2ind)
        self.test = convert_data_to_index_form(self.test, self.word2ind)
        self.val = convert_data_to_index_form(self.val, self.word2ind)

        self.voc_size = len(self.word2ind)

    def sample_pairs_from_train(self, win_size):
        s = np.random.choice(self.train, 1)[0]

        inp_ind = np.random.randint(len(s))

        return get_word_pairs(s, inp_ind, win_size)

    def get_batch_data(self, batch_size, win_size):
        batch_data = []

        while len(batch_data) < batch_size:
            batch_data += self.sample_pairs_from_train(win_size)

        return batch_data[: batch_size]
