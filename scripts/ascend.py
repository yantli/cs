# trying out ASCEND

# 08/19/2023

import csv
import re
import datasets
dataset = datasets.load_dataset("CAiRE/ASCEND")

# collecting all the cs sentences from 'train', 'test' and 'validation'
def get_sents(part_of_dataset):
    sents = []
    n = 0 
    while n < len(dataset[part_of_dataset]):
        if dataset[part_of_dataset][n].get('language') == 'mixed':
            sent = dataset[part_of_dataset][n].get('transcription')
            topic = dataset[part_of_dataset][n].get('topic')
            sents.append((sent, topic))
        n += 1
    return sents

def save_csv(output_file, list_to_save):
        with open(output_file, 'a', newline='') as csvf:
            writer = csv.writer(csvf, delimiter = ',')
            for item in list_to_save:
                writer.writerow(item)

def create_extended_corpus(allsents):
    newallsents = []
    # add cs_word_length to each line (for nonCS line, it will just be 0)
    eng_pattern = re.compile(r'[a-zA-Z]+')
    n = 0
    while n < len(allsents):
        sent = allsents[n][0]
        cs_word_end = eng_pattern.search(sent).span()[1]
        if not eng_pattern.search(sent[cs_word_end+1:]):
            newsent = (allsents[n][0], allsents[n][1],'1')
        else:
            newsent = (allsents[n][0], allsents[n][1],'more')
        newallsents.append(newsent)
        n += 1
    
    return newallsents

def collect_mono_cs_words(newallsents):
    cs_words = []
    eng_pattern = re.compile(r'[a-zA-Z]+')
    for sent in newallsents:
        if sent[2] == '1':
            cs_index = eng_pattern.search(sent[0]).span()
            cs_word = sent[0][int(cs_index[0]):int(cs_index[1])]
            if cs_word not in cs_words:
                cs_words.append(cs_word)
    
    return cs_words


if __name__ == "__main__":
    trainsents = get_sents('train')
    testsents = get_sents('test')
    valsents = get_sents('validation')
    allsents = trainsents + testsents + valsents
    newallsents = create_extended_corpus(allsents)
    save_csv('/Users/yanting/Desktop/cs/data/ascend.csv', newallsents)