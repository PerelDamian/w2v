import string
import re
import numpy as np
import itertools

alphanumeric_ascii = list(filter(str.isalnum, string.printable))


def get_word_pairs(s, inp_ind, win_size):
    min_context_ind = max(0, inp_ind - win_size)
    max_context_ind = min(len(s), inp_ind + win_size + 1)
    context_indices = range(min_context_ind, max_context_ind)
    input_word = s[inp_ind]
    context_words = [s[context_ind] for context_ind in context_indices if context_ind != inp_ind]
    pairs = [(input_word, context_word) for context_word in context_words]

    return pairs


def get_sentence_pairs(sentence, win_size):
    pairs = list(itertools.chain(*(get_word_pairs(sentence, inp_ind, win_size) for inp_ind in range(len(sentence)))))

    return pairs


def get_data_pairs(sentences_list, win_size, max_pairs=np.inf):
    pairs = []

    for sentence in sentences_list:
        pairs += get_sentence_pairs(sentence, win_size)
        if len(pairs) > max_pairs:
            break

    return pairs


def clean_sentence(sentence):
    sentence = sentence.lower()

    # remove punctuation. it's better to replace with spaces and then convert multiple spaces to one.
    for ch in string.punctuation:
        sentence = sentence.replace(ch, ' ')
    sentence = re.sub('\s+', ' ', sentence).strip()

    # keep only alphanumeric chars
    sentence = ''.join(letter for letter in sentence if letter in alphanumeric_ascii or letter == ' ')

    # delete words with less than 3 chars
    sentence = [word for word in sentence.split(' ') if len(word) >= 3]

    return sentence


def convert_data_to_index_form(data, word2ind):
    data_index_form = []

    for sentence in data:
        s_ind_form = []
        for word in sentence:
            try:
                ind = word2ind[word]
            except KeyError:
                ind = word2ind['<unk>']
            s_ind_form.append(ind)
        if len(sentence) > 1:
            data_index_form.append(s_ind_form)

    return data_index_form


class SentenceSplitter:
    def __init__(self, dataset_split_path):
        with open(dataset_split_path) as f:
            f.readline()  # skip header
            lines = f.readlines()
        lines = list(map(lambda line: line.strip().split(','), lines))
        self.train_indices = [line[0] for line in lines if line[1] == "1"]
        self.test_indices = [line[0] for line in lines if line[1] == "2"]
        self.val_indices = [line[0] for line in lines if line[1] == "3"]

    def get_set_from_sentence_index(self, sentence_ind):
        if sentence_ind in self.train_indices:
            return "train"
        elif sentence_ind in self.test_indices:
            return "test"
        elif sentence_ind in self.val_indices:
            return "validation"
        else:
            raise KeyError


def get_train_vocab_dicts(data):
    voc_list = ['<unk>'] + list(set(word for sentence in data for word in sentence))
    voc_indices = range(len(voc_list))

    return dict(zip(voc_list, voc_indices)), dict(enumerate(voc_list))


class DataParser:
    def __init__(self, sentences_path, splitter):
        with open(sentences_path) as f:
            f.readline()  # skip header
            lines = f.readlines()

        lines = [line.strip().split('\t') for line in lines]
        index_to_sentence = {line[0]: clean_sentence(line[1]) for line in lines}

        self.train = []
        self.test = []
        self.val = []

        for ind, sentence in index_to_sentence.items():
            set_ind = splitter.get_set_from_sentence_index(ind)
            if set_ind == "train":
                self.train.append(sentence)
            elif set_ind == "test":
                self.test.append(sentence)
            elif set_ind == "validation":
                self.val.append(sentence)

        self.word2ind, self.ind2word = get_train_vocab_dicts(self.train)

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
