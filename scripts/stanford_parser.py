# segmenting and tagging the sentences in ASCEND corpus

# 10/16/2023

import csv
from multiprocessing.reduction import DupFd
import re

# keeping only the sentences, discarding sentence numbers and themes/topics
def get_pure_sents(input_file):
    puresents = []
    fullsents = []
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            puresents.append(line[0])
            fullsents.append(line)
    return puresents, fullsents

# saving the sentences into a .txt file
def save_txt(output_file, sent_list):
    with open(output_file, 'w') as f:
        for item in sent_list:
            f.write("{}\n".format(item))

# reading in the raw tagged .tsv file with format like:
# 我      PN
# 刚刚    AD
# 开始    VV
# record  NR
# this is to change it to (sentence_ID, 我 刚刚 开始 record, PN AD VV NR)

def processing_tagged_raw(tagged_file):    
    sentences = []
    with open(tagged_file, 'r', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        sent_count = 1
        sent = []
        for word in tsv_reader:
            if len(word) > 0:
                sent.append(word)
            else:
                segmented_sent = ' '.join([word[0] for word in sent])
                pos_tags = ' '.join([word[1] for word in sent])
                sent_and_tag = (sent_count, segmented_sent, pos_tags)
                sentences.append(sent_and_tag)
                sent_count += 1
                sent = []
            
    return sentences

# now integrate this information into ascend.csv, creating a file that has these columns:
# sentence_ID, segmented_sent, POS_tags, num_of_CS_words, CS_word, idx_of_CS_word, POS_of_CS, topic, originalset, originalID
# num_of_CS_words: any integer or "more"
# CS_word: tuple if the sentence has more than one seperated CS words, blank if the sentence has consecutive CS words
# idx_of_CS_word: tuple if the sentence has more than one seperated CS words, blank if the sentence has consecutive CS words
# POS_of_CS: tuple if the sentence has more than one seperated CS words, blank if the sentence has consecutive CS words
def create_extended_corpus(original_corpus, tagged_file):
    newsents = []
    original_sentences = get_pure_sents(original_corpus)[1]
    tagged_sentences = processing_tagged_raw(tagged_file)
    eng_pattern = re.compile(r'[a-zA-Z]+')
    char_pattern = re.compile(r'[\u4e00-\u9fff]')
    i = 0
    while i < len(tagged_sentences):
        sentence_ID = tagged_sentences[i][0]
        segmented_sent = tagged_sentences[i][1]
        POS_tags = tagged_sentences[i][2]
        num_of_CS_words = original_sentences[i][1]
        topic = original_sentences[i][2]
        originalset = original_sentences[i][3]
        originalID = original_sentences[i][4]
        if num_of_CS_words != 'more':
            idx_of_CS_word = []
            POS_of_CS = []
            CS_word = []
            for n in range(len(segmented_sent.split(' '))):
                if eng_pattern.search(segmented_sent.split(' ')[n]) and not char_pattern.search(segmented_sent.split(' ')[n]):
                    idx_of_CS_word.append(n)
                    POS_of_CS.append(POS_tags.split(' ')[n])
                    CS_word.append(segmented_sent.split(' ')[n])
        else:
            CS_word = None
            idx_of_CS_word = None
            POS_of_CS = None
        newsent = (sentence_ID, segmented_sent, POS_tags, num_of_CS_words, CS_word, idx_of_CS_word, POS_of_CS, topic, originalset, originalID)
        newsents.append(newsent)
        i += 1

    return newsents

def save_csv(output_file, list_to_save):
    with open(output_file, 'a', newline='') as csvf:
        writer = csv.writer(csvf, delimiter = ',')
        for item in list_to_save:
            writer.writerow(item)

if __name__ == "__main__":
    ascend_cs_corpus_path = "/Users/yanting/Desktop/cs/data/ascend_cs_full.csv"
    # puresentences_cs = get_pure_sents(ascend_cs_corpus_path)[0]
    # save_txt("/Users/yanting/Desktop/cs/data/ascend_puresents.txt", puresentences_cs)
    tagged_cs_file_path = "/Users/yanting/Desktop/cs/data/ascend_cs_full.tag"
    expanded_cs_corpus = create_extended_corpus(ascend_cs_corpus_path, tagged_cs_file_path)
    save_csv("/Users/yanting/Desktop/cs/data/ascend_cs_full_corpus.csv", expanded_cs_corpus)

    ascend_zh_corpus_path = "/Users/yanting/Desktop/cs/data/ascend_monozh.csv"
    puresentences_zh = get_pure_sents(ascend_zh_corpus_path)[0]
    save_txt("/Users/yanting/Desktop/cs/data/ascend_monozh_puresents.txt", puresentences_zh)
    tagged_zh_file_path = "/Users/yanting/Desktop/cs/data/ascend_monozh.tag"
    