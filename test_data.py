from data_utils import clean_sentence, SentenceSplitter, DataParser


def test_cleaning():
    assert clean_sentence("avi, biter,is,!$%*an asshole") == ['avi', 'biter', 'asshole']
    assert clean_sentence("avi       biter,") == ['avi', 'biter']


spltr = SentenceSplitter(r'data/datasetSplit.txt')
data = DataParser(r'data/datasetSentences.txt', spltr)


def test_sentence_splitter():
    assert spltr.get_set_from_sentence_index(8263) == 2
    assert spltr.get_set_from_sentence_index(8200) == 1
    assert spltr.get_set_from_sentence_index(7443) == 3
    assert all([type(ind) == int for ind in spltr.train_indices])
    assert all([type(ind) == int for ind in spltr.test_indices])
    assert all([type(ind) == int for ind in spltr.val_indices])
