# run the same experiment and analysis based on the expanded ASCEND dataset that contains tagged sentences

# 10/16/2023

import re
import csv
from curses import pair_content
from time import monotonic
from googletrans import Translator
import hanzidentifier
import sys
sys.path.append('/Users/yanting/Desktop/cs/scripts/')
from info_collector import create_KDtree, calculate_cosine_similarity
from stanford_parser import processing_tagged_raw

def save_whole_file(dataset_file):
    whole_file = []
    with open(dataset_file, 'r', encoding = 'utf-8') as f:
        filereader = csv.reader(f, delimiter = ',')
        for row in filereader:
            whole_file.append(row)

    return whole_file

def word_collector(dataset_file):
    whole_file = save_whole_file(dataset_file)
    CSall = []
    CSall_tagged = []
    CSnoun = []
    nonCSall = []
    nonCSall_tagged = []
    nonCSnoun = []
    eng_pattern = re.compile(r'[a-zA-Z]+')
    char_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    n = 0
    while n < len(whole_file):
        words = whole_file[n][1].split(' ')
        tags = whole_file[n][2].split(' ')

        # getting CS words
        if whole_file[n][3] != 'more':
            if whole_file[n][3] == '1':
                cs_word = [whole_file[n][4][2:-2].lower()]
                cs_pos = [whole_file[n][6][2:-2]]
                pair = [cs_word[0] + '-' + cs_pos[0]]
            else:
                    words = whole_file[n][4][1:-1].split(', ')
                    pos = whole_file[n][6][1:-1].split(', ')
                    cs_word = [item[1:-1].lower() for item in words]
                    cs_pos = [item[1:-1] for item in pos]
                    pair = [cs_word[i] + '-' + cs_pos[i] for i in range(len(cs_word))]
            for i in range(len(cs_word)):
                if cs_word[i] not in CSall:
                    CSall.append(cs_word[i])
                if pair[i] not in CSall_tagged:
                    CSall_tagged.append(pair[i])
                if cs_pos[i] == 'NN' and cs_word[i] not in CSnoun:
                    CSnoun.append(cs_word[i])
            
        # getting non-CS words
        i = 0
        while i < len(words):
            if char_pattern.search(words[i]) and not eng_pattern.search(words[i]):
                nonCS_pair = words[i] + '-' + tags[i]
                if words[i] not in nonCSall:
                    nonCSall.append(words[i])
                if nonCS_pair not in nonCSall_tagged:
                    nonCSall_tagged.append(nonCS_pair)
                if tags[i] == 'NN' and words[i] not in nonCSnoun:
                    nonCSnoun.append(words[i])
            i += 1

        n += 1
    
    return CSall, CSall_tagged, CSnoun, nonCSall, nonCSall_tagged, nonCSnoun

def get_monozh_words(monozh_tagged_file):
    monozhwords = []
    monozhNN = []
    tagged_zh_sents = processing_tagged_raw(monozh_tagged_file)
    for item in tagged_zh_sents:
        words = item[1].split()
        pos = item[2].split()
        for i in range(len(words)):
            if pos[i] == 'NN' and words[i] not in monozhNN:
                monozhNN.append(words[i])
            if words[i] not in monozhwords:
                monozhwords.append(words[i])
    
    return monozhwords, monozhNN

# to avoid error, try feeding shorter word_lists, there might be a quota per day
def get_single_noncs_words(word_list):
    nonCS_dict = {}
    translator = Translator()

    vocab = [word for word in word_list if hanzidentifier.is_simplified(word) and word != '、']
    for word in vocab:
        if len(translator.translate(word).text.split()) == 1 and word not in nonCS_dict.keys():
            nonCS_dict[word] = translator.translate(word).text.lower()
    
    eng_word_pool = [word for word in set(nonCS_dict.values())]

    return eng_word_pool

def calculate_vec_distances(word_list, word_vectors_array, kdtree, passed_in_words):
    results = []
    char_pattern = re.compile(r'[\u4e00-\u9fff]')
    katakana_pattern = re.compile(r'[\u30a0-\u30ff]')
    hiragana_pattern = re.compile(r'[\u3040-\u309f]')

    for word in passed_in_words:
        if word in word_list:
            index = word_list.index(word)
            query_vector = word_vectors_array[index]
            distances, indices = kdtree.query(query_vector, k=10000)
            n = 0
            while n < len(indices):
                if bool(char_pattern.search(word_list[indices[n]])) and not (katakana_pattern.search(word_list[indices[n]]) or hiragana_pattern.search(word_list[indices[n]])) and hanzidentifier.is_simplified(word_list[indices[n]]):
                    matching_word = word_list[indices[n]]
                    distance = distances[n]
                    matching_word_index = word_list.index(matching_word)
                    matching_word_query_vector = word_vectors_array[matching_word_index]
                    cos_sim = calculate_cosine_similarity(query_vector, matching_word_query_vector)
                    result = (word, matching_word, distance, n, cos_sim)
                    results.append(result)
                    break
                else:
                    n += 1
            if n == len(indices):
                result = (word, 'need more vectors')
                results.append(result)
        else:
            result = (word)
            results.append(result)

    return results

def save_results(output_file, list_to_save):
    with open(output_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        for item in list_to_save:
            if len(item) == 5 and not isinstance(item[2], str) and not isinstance(item[4], str):
                writer.writerow(item)
            else:
                writer.writerow([item])

if __name__ == "__main__":
    dataset_path = '/Users/yanting/Desktop/cs/data/ascend_cs_full_corpus.csv'
    shared_vec_path = '/Users/yanting/Desktop/cs/word_vector/shared_30k_engzh.vec'
    tagged_zh_file_path = "/Users/yanting/Desktop/cs/data/ascend_monozh.tag"
    word_list, word_vectors_array, kdtree = create_KDtree(shared_vec_path)
    
    CSall, CSall_tagged, CSnoun, nonCSall, nonCSall_tagged, nonCSnoun = word_collector(dataset_path)
    all_eng_word_pool, all_eng_noun_pool = get_single_noncs_words(dataset_path)

    csNouns_results = calculate_vec_distances(word_list, word_vectors_array, kdtree, CSnoun)
    save_results('/Users/yanting/Desktop/cs/results/ascend_csNNSingleMulti_closest_zhneighbor_simp.csv', csNouns_results)

    
