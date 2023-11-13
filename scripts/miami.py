# trying out Miami
# documentation: http://bangortalk.org.uk/docs/Miami_doc.pdf

# 10/9/2023

import csv
import re

# collecting all the cs sentences
def read_original_file(original_tsv_file):
    sents = []
    with open(original_tsv_file) as file:
        rd = csv.reader(file, delimiter="\t", quotechar='"')
        for row in rd:
            sents.append(row)
    return sents[1:-1]

def sort_sents(original_tsv_file):
    allsents = read_original_file(original_tsv_file)
    cs_sents = []
    non_cs_sents = []
    undefined_sents = []
    total_num_of_utt = int(allsents[-1][1])
    n = 1
    while n < total_num_of_utt:
        utt = [entry for entry in allsents if entry[1]==str(n)]
        uttlang = [entry[9] for entry in allsents if entry[1]==str(n)]
        if 'spa' in uttlang and 'eng' in uttlang:
            cs_sents.append(utt)
        elif 'eng&spa+eng' in utt or 'eng&spa' in utt:
            undefined_sents.append(utt)
        else:
            non_cs_sents.append(utt)
        n += 1
    
    return cs_sents, non_cs_sents, undefined_sents






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
    herring1 = ''
    trainsents = get_sents('train')
    testsents = get_sents('test')
    valsents = get_sents('validation')
    allsents = trainsents + testsents + valsents
    newallsents = create_extended_corpus(allsents)
    save_csv('/Users/yanting/Desktop/cs/data/ascend.csv', newallsents)


# the original file has the following info:
# word_id: 1 
# utterance_id: 1    
# location: 1           this is where the word is located in the utterance       
# surface: well         this is the word
# auto: well.ADV        this is the word with POS tag
# fix:      
# eng:     
# com: 
# speaker: CHL          info of the speaker is provided on the corpus website
# langid: eng           this indicate what language the word is in: 'eng', 'spa', 'eng&spa'(undetermined), 'spa+eng'(word with first morpheme(s) Spanish, final morpheme(s) English), 'eng+spa'(word with first morpheme(s) English, final morpheme(s) Spanish), '999'(for punctuation)
# filename: herring1
# clause:  
# clauseno:
                                