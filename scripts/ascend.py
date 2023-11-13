# extracting information from the original ASCEND corpus
# enter the environment ASCEND before running the script

# 08/19/2023

import csv
import re
import datasets
dataset = datasets.load_dataset("CAiRE/ASCEND")

eng_pattern = re.compile(r'[a-zA-Z]+')
char_pattern = re.compile(r'[\u4e00-\u9fff]')

# collecting all the cs sentences from 'train', 'test' and 'validation'
# for cs sents, lang = 'mixed'; for zh sents, lang = 'zh'; for eng sents, lang = 'en'
def get_sents(part_of_dataset, lang):
    sents = []
    n = 0 
    while n < len(dataset[part_of_dataset]):
        if dataset[part_of_dataset][n].get('language') == lang:
            sent = dataset[part_of_dataset][n].get('transcription')
            if sent.find('[UNK]') and sent.find('[UNK]') != 0:
                sent = ''.join(sent.split(' [UNK]'))
            elif sent.find('[UNK]') and sent.find('[UNK]') == 0:
                sent = sent[6:]
            topic = dataset[part_of_dataset][n].get('topic')
            sents.append((sent, topic, part_of_dataset, n))
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
    n = 0
    while n < len(allsents):
        sent = allsents[n][0]
        cs_word_end = eng_pattern.search(sent).span()[1]
        remaining_sent = sent[cs_word_end:]
        cs_count = cs_word_counter(remaining_sent, 1)
        if cs_count > 10000:
            cs_count = 'more'
        newsent = (allsents[n][0], cs_count, allsents[n][1], allsents[n][2], allsents[n][3])
        newallsents.append(newsent)
        n += 1
    
    return newallsents

def cs_word_counter(sentence, current_cs_count):
    if not eng_pattern.search(sentence):
        cs_count = current_cs_count
    else:
        if eng_pattern.search(sentence).span()[0] == 0:
            cs_count = 10001
        elif eng_pattern.search(sentence).span()[0] == 1 and sentence[0] == ' ':
            cs_count = 10001
        else:
            cs_word_end = eng_pattern.search(sentence).span()[1]
            if cs_word_end == len(sentence):
                cs_count = current_cs_count + 1
            else:
                cs_count = cs_word_counter(sentence[cs_word_end:], current_cs_count+1)

    return cs_count

def collect_mono_cs_words(newallsents):
    cs_words = []
    for sent in newallsents:
        sentence = sent[0]
        if sent[2] == 1:
            cs_index = eng_pattern.search(sentence).span()
            cs_word = sentence[int(cs_index[0]):int(cs_index[1])]
            if cs_word.lower() not in cs_words:
                cs_words.append(cs_word.lower())
        elif type(sent[2]) == int and sent[2] > 1:
            n = 0 
            while n < sent[2]:
                cs_index = eng_pattern.search(sentence).span()
                cs_word = sentence[int(cs_index[0]):int(cs_index[1])]
                if cs_word.lower() not in cs_words:
                    cs_words.append(cs_word.lower())
                sentence = sentence[cs_index[1]:]
                n += 1
    
    return cs_words

def save_txt(output_file, list_to_save):
    with open(output_file, 'w') as output:
        for row in list_to_save:
            output.write(row + '\n')

if __name__ == "__main__":
    trainsents = get_sents('train','mixed')
    testsents = get_sents('test','mixed')
    valsents = get_sents('validation','mixed')
    allsents = trainsents + testsents + valsents
    newallsents = create_extended_corpus(allsents)
    save_csv('/Users/yanting/Desktop/cs/data/ascend_cs.csv', newallsents)
    puresents = [sent[0] for sent in newallsents]
    save_txt('/Users/yanting/Desktop/cs/data/ascend_cs.txt', puresents)