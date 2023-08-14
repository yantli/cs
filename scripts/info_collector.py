# reading in raw data and adding info about:
# 1) English word freq
# 2) English & Chinese word length
# 3) English & Chinese word vector 

# 04/23/2023

import csv
import chinese_converter # https://pypi.org/project/chinese-converter/ 
import pinyin   # https://pypi.org/project/pinyin/
import math
import numpy as np
import re
from googletrans import Translator
from scipy.spatial import KDTree
import pickle
import hanzidentifier
import nltk
from nltk.corpus import brown
import random


def save_whole_file(original_file_path):
    whole_file = []
    cs_lines = []
    non_cs_lines = []
    with open(original_file_path, 'r', encoding = 'utf-8') as f:
        filereader = csv.reader(f, delimiter = ',')
        for row in filereader:
            whole_file.append(row)
            all_lines = whole_file[1:]
            n = 0
        while n < len(whole_file)-2:
            cs_lines.append(all_lines[n])
            non_cs_lines.append(all_lines[n+1])
            n += 2

    return all_lines, cs_lines, non_cs_lines
      
def word_collector(original_file_path):
    eng_words = []
    trans_words = []
    zh_words = []
    POS_trans = []
    with open(original_file_path, 'r', encoding = 'utf-8') as f:
        filereader = csv.reader(f, delimiter = ',')
        for row in filereader:
            if row[0] == "code-switch":
                eng_word = row[7]
                eng_words.append(eng_word)
                trans_word = row[8]
                trans_words.append(trans_word)
            elif row[0] == "non-code-switch":
                zh_word = row[8]
                zh_words.append(zh_word)

    # translator = Translator()
    # zh_words_trans = [translator.translate(word).text for word in zh_words]    
    # zh_words_trans = [word.lower() for word in zh_words_trans]
    
    # return eng_words, trans_words, zh_words, zh_words_trans
    return eng_words, trans_words, zh_words, POS_trans

def create_KDtree(word_vec_path):
    word_vectors = []
    word_list = []
    with open(word_vec_path,'r', encoding='utf-8') as vecfile:
        for line in vecfile:
            if len(line.split())>300:
                vector = np.array(line.split()[-300:], dtype=np.float32)
                word_vectors.append(vector)

                vec_idx = line.index(' '.join(line.split()[-300:]))
                word = line[:vec_idx-1]
                word_list.append(word)
                # print(word)
    
    word_vectors_array = np.array(word_vectors)
    kdtree = KDTree(word_vectors_array)

    return word_list, word_vectors_array, kdtree

def save_kdtree(word_vec_path, output_file):
    kdtree = create_KDtree(word_vec_path)[2]
    with open(output_file, 'wb') as f:
        pickle.dump(kdtree, f)

# when lang == "eng" or "zh", we are using monolingual kdtrees
# when lang == "mixed_closest1", we are using combined kdtrees of both languages, and we are finding the closest vector in language B for vectors in language A
# when lang == "mixed_closestcir", we are using combined kdtrees of both languages, and we are finding how many vectors in language B there are within a certain distance from the vector in language A
# when lang == "mixed_ptp" (i.e. point-to-point), we are using combined kdtrees of both languages, and we are calculating the difference between the CS word pair
# when passed_in_words is True, it means that we are passing in a specified list of words instead of using the lists from word_collector(original_file_path)
def calculate_vec_distance(word_list, word_vectors_array, kdtree, lang, passed_in_words = None):
    results = []
    char_pattern = re.compile(r'[\u4e00-\u9fff]')
    katakana_pattern = re.compile(r'[\u30a0-\u30ff]')
    hiragana_pattern = re.compile(r'[\u3040-\u309f]')

    if passed_in_words is None:
        eng_words = word_collector(original_file_path)[0]
        trans_words = word_collector(original_file_path)[1]
        traditional_trans_words = [chinese_converter.to_traditional(word) for word in trans_words]
        zh_words = word_collector(original_file_path)[2]
        traditional_zh_words = [chinese_converter.to_traditional(word) for word in zh_words]
        
        uniq_eng_words = []
        for word in eng_words:
            if word not in uniq_eng_words:
                uniq_eng_words.append(word)

        uniq_zh_words = []
        for word in zh_words:
            if word not in uniq_zh_words:
                uniq_zh_words.append(word)
        
        uniq_traditional_zh_words = [chinese_converter.to_traditional(word) for word in uniq_zh_words]

        if lang == 'eng':
            for word in uniq_eng_words:
                if word in word_list:
                    index = word_list.index(word)
                    # print(index)
                    query_vector = word_vectors_array[index]
                    distances, indices = kdtree.query(query_vector, k=11)
                    # print(distances)
                    average_distance = sum(distances)/(len(distances)-1)
                    # print(average_distance)
                    result = (word, index, average_distance, [index for index in indices])
                    # print(result)
                else:
                    result = (word)
                results.append(result)

        if lang == 'zh':
            for word in uniq_zh_words:
                if word in word_list:
                    index = word_list.index(word)
                    query_vector = word_vectors_array[index]
                    distances, indices = kdtree.query(query_vector, k=11)
                    average_distance = sum(distances)/(len(distances)-1)
                    result = (word, index, average_distance, [index for index in indices], 'simplified')
                elif traditional_zh_words[zh_words.index(word)] in word_list:
                    index = word_list.index(traditional_zh_words[zh_words.index(word)])
                    query_vector = word_vectors_array[index]
                    distances, indices = kdtree.query(query_vector, k=11)
                    average_distance = sum(distances)/(len(distances)-1)
                    result = (word, index, average_distance, [index for index in indices], traditional_zh_words[zh_words.index(word)])
                else:
                    result = (word)
                results.append(result)                
        
        if lang == 'mixed_closest1':
            for word in uniq_eng_words:
                if word in word_list:
                    index = word_list.index(word)
                    # print(index)
                    query_vector = word_vectors_array[index]
                    distances, indices = kdtree.query(query_vector, k=5001)
                    n = 0 
                    while n < len(indices):
                        if bool(char_pattern.search(word_list[indices[n]])) and not (katakana_pattern.search(word_list[indices[n]]) or hiragana_pattern.search(word_list[indices[n]])):
                            matching_word = word_list[indices[n]]
                            distance = distances[n]
                            result = (word, matching_word, distance, n)
                            results.append(result)
                            break
                        else:
                            n += 1
                    if n == len(indices):
                        result = (word, "need more vectors")
                        results.append(result)
                        # print(result)
                else:
                    result = (word)
                    results.append(result)

        if lang == 'mixed_closestcir':
            for word in uniq_eng_words:
                if word in word_list:
                    index = word_list.index(word)
                    query_vector = word_vectors_array[index]
                    distance_threshold = 1.1
                    indices = kdtree.query_ball_point(query_vector, distance_threshold)
                    ZH_neighbors = []
                    for index in indices:
                        if bool(char_pattern.search(word_list[index])) and not (katakana_pattern.search(word_list[index]) or hiragana_pattern.search(word_list[index])):
                            ZH_neighbors.append(index)
                    result = (word, len(ZH_neighbors), [word_list[n] for n in ZH_neighbors])
                    # print(result)
                    results.append(result)
                else:
                    result = (word)
                    results.append(result)
                    
        if lang == 'mixed_ptp':
            for n in range(len(eng_words)):
                engword = eng_words[n]
                transwordsimp = trans_words[n]
                transwordtradi = traditional_trans_words[n]
                if engword in word_list and (transwordsimp in word_list or transwordtradi in word_list):
                    if transwordsimp in word_list:
                        transword = transwordsimp
                    else:
                        transword = transwordtradi
                    eng_index = word_list.index(engword)
                    eng_vec = word_vectors_array[eng_index]
                    trans_index = word_list.index(transword)
                    trans_vec = word_vectors_array[trans_index]
                    distance = vector_distance(eng_vec, trans_vec)
                    result = (engword, transword, distance)
                    # print(result)
                    results.append(result)
                else:
                    result = (engword)
                    results.append(result)

    else:
        for word in passed_in_words:
            if word in word_list:
                index = word_list.index(word)
                query_vector = word_vectors_array[index]
                distances, indices = kdtree.query(query_vector, k=5000)
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

# def eng_vector_finder(original_file_path, eng_word_vec_path):
#     eng_words = word_collector(original_file_path)[0]
#     eng_vec_dict = {}
#     with open(eng_word_vec_path,'r', encoding='utf-8') as vecfile:
#         line = 'xxx'
#         while line:
#             line = vecfile.readline()
#             if len(line.split())>0:
#                 num_str = re.findall(r'-?\d+\.\d+|-?\d+', line)
#                 line_as_list = line.split()
#                 wordlist = [i for i in line_as_list if i not in num_str]
#                 word = ' '.join(wordlist)
                
#                 if word in eng_words and word not in eng_vec_dict.keys():
#                     numbers = [float(n) for n in num_str]
#                     vec = np.array(numbers)
#                     eng_vec_dict[word] = vec
# 
#     return eng_vec_dict

# compared to eng_vector_finder, this one is easier and faster, 
# but it takes in "the", " the", "the ", " the ", etc.
# and it can only work on one-word English words
# eng_vec_dict_easy contains values that is the whole word vector entry as a str
# uniq_eng_vec_dict contains values that is the number-only vector that is ready for cos sim calculation
def eng_vector_finder_easy(original_file_path, eng_word_vec_path):
    eng_words = word_collector(original_file_path)[0] + word_collector(original_file_path)[3]
    eng_words = list(set(eng_words))
    eng_vec_dict_easy = {}
    with open(eng_word_vec_path,'r', encoding='utf-8') as vecfile:
        for line in vecfile:
            if len(line.split())>0 and line.split()[0] in eng_words:
                word = line.split()[0]
                if word not in eng_vec_dict_easy.keys():
                    eng_vec_dict_easy[word] = [line]
                else:
                    eng_vec_dict_easy[word].append(line)
            
        allvec = []
        for value in eng_vec_dict_easy.values():
            for vec in value:
                allvec.append(vec)

        uniq_eng_vec_dict = {}
        for vec in allvec:
            # assumption: each vector contains exactly 300 numbers 
            num_str = vec.split()[-300:]
            num_list = [float(n) for n in num_str]
            numbers = np.array(num_list)

            num_idx = vec.index(' '.join(num_str))
            word = vec[:num_idx-1]
            if word in eng_words and word not in uniq_eng_vec_dict.keys():
                uniq_eng_vec_dict[word] = numbers

    return eng_vec_dict_easy, uniq_eng_vec_dict

def zh_vector_finder(original_file_path, zh_word_vec_path):
    zh_words = word_collector(original_file_path)[1] + word_collector(original_file_path)[2]
    zh_words = list(set(zh_words))
    traditional_zh_words = [chinese_converter.to_traditional(word) for word in zh_words]
    # creating a dictionary for the traditional forms of the simplified words. 
    # The two forms might be the same for some words though and the entry is kept here anyways.
    simp_tradi_dict = {}
    for i in range(len(zh_words)):
        simp_tradi_dict[zh_words[i]] = traditional_zh_words[i]
    
    zh_vec_dict = {}
    with open(zh_word_vec_path,'r', encoding='utf-8') as vecfile:
        for line in vecfile:
            if len(line.split()) > 0 and line.split()[0] in zh_words:
                word = line.split()[0]
                if word not in zh_vec_dict.keys():
                    zh_vec_dict[word] = [line]
                else:
                    zh_vec_dict[word].append(line)

            # if the vector is for a traditional word but its simplified version is not yet in the dict:        
            elif len(line.split()) > 0 and line.split()[0] in traditional_zh_words and chinese_converter.to_simplified(line.split()[0]) not in zh_vec_dict.keys():
                word = line.split()[0]
                if word not in zh_vec_dict.keys():
                    zh_vec_dict[word] = [line]
                else:
                    zh_vec_dict[word].append(line)

        # The zh_vec_dict may have words with both forms. We want to keep the simplified key only.        
        uniq_zh_vec_dict = {}
        duplicated = []
        traditional_vector = []
        for simp, tradi in simp_tradi_dict.items():
            if simp in zh_vec_dict.keys() and tradi in zh_vec_dict.keys() and simp != tradi:
                duplicated.append(tradi)
        for key, value in zh_vec_dict.items():
            if key not in duplicated:
                num_str = value[0].split()[-300:]
                num_list = [float(n) for n in num_str]
                numbers = np.array(num_list)

                num_idx = value[0].index(' '.join(num_str))
                word = value[0][:num_idx-1]

                if word in zh_words and word not in uniq_zh_vec_dict.keys():
                    uniq_zh_vec_dict[word] = numbers
                elif word in traditional_zh_words and word not in uniq_zh_vec_dict.keys():
                    uniq_zh_vec_dict[chinese_converter.to_simplified(word)] = numbers
                    traditional_vector.append(word)

    return uniq_zh_vec_dict

def vec_dict_saver(vec_dict, output_file):
    with open(output_file, 'w', encoding = 'utf-8') as f:
        for word, vec in vec_dict.items():
            line = (word, str(vec),'\n')
            f.writelines(line)

def calculate_cosine_similarity(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_similarity

def gather_cos_similarity(original_file_path, eng_word_vec_path, zh_word_vec_path):
    cs_lines = save_whole_file(original_file_path)[1]
    non_cs_lines = save_whole_file(original_file_path)[2]
    zh_words_trans = word_collector(original_file_path)[3]
    eng_vec_dict = eng_vector_finder_easy(original_file_path, eng_word_vec_path)[1]
    zh_vec_dict = zh_vector_finder(original_file_path, zh_word_vec_path)
    
    cos_sim_list_cswords = []
    for line in cs_lines:
        eng_word = line[7]
        trans_word = line[8]
        sent_id = line[2]
        if eng_word in eng_vec_dict.keys() and trans_word in zh_vec_dict.keys():
            eng_wordvec = eng_vec_dict.get(eng_word)
            trans_wordvec = zh_vec_dict.get(trans_word)
            cos_sim = calculate_cosine_similarity(eng_wordvec, trans_wordvec)
            entry = (sent_id, eng_word, trans_word, cos_sim)
            cos_sim_list_cswords.append(entry)

    cos_sim_list_noncswords = []
    for i in range(len(non_cs_lines)):
        zh_word = non_cs_lines[i][8]
        eng_trans = zh_words_trans[i]
        sent_id = non_cs_lines[i][2]
        if eng_trans in eng_vec_dict.keys() and zh_word in zh_vec_dict.keys():
            eng_wordvec = eng_vec_dict.get(eng_trans)
            zh_wordvec = zh_vec_dict.get(zh_word)
            cos_sim = calculate_cosine_similarity(eng_wordvec, zh_wordvec)
            entry = (sent_id, zh_word, eng_trans, cos_sim)
            cos_sim_list_noncswords.append(entry)

    return cos_sim_list_cswords, cos_sim_list_noncswords

def save_csv_w(output_file, list_to_save):
    with open(output_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerows(list_to_save)

def save_csv_a(output_file, tuple_to_save):
        with open(output_file, 'a', newline='') as csvf:
            writer = csv.writer(csvf, delimiter = ',')
            writer.writerow(tuple_to_save)     

def vector_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def eng_wordlen_calculator(original_file_path):
    eng_words = word_collector(original_file_path)[0]
    eng_wordlen_dict = {}
    for word in eng_words:
        eng_wordlen_dict[word] = len(word)
    return eng_wordlen_dict

def trans_wordlen_calculator(original_file_path):
    trans_words = word_collector(original_file_path)[1]
    pinyinlen_dict = {}
    charlen_dict = {}
    for word in trans_words:
        pinyinlen_dict[word] = len(pinyin.get(word, format="strip", delimiter=""))
        charlen_dict[word] = len(word)
    return pinyinlen_dict, charlen_dict

def eng_word_freq_generator(original_file_path, corpus_file_path):
    word_dict = {}
    with open(corpus_file_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            print(line.split())
   
    freq_dict = {}

    return freq_dict

def get_random_words_nltk(word_type, num_to_pick):
    top_list = []
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    corpus = brown.words()
    word_list = [word for (word, pos) in nltk.pos_tag(corpus) if pos.startswith(word_type)]
    top_words = nltk.FreqDist(word_list).most_common(num_to_pick)
    for word, frequency in top_words:
        top_list.append(word)

    return top_list

def get_random_words_corpus():
    non_cs_lines = save_whole_file(original_file_path)[2]
    
    all_noncs_words = []
    for line in non_cs_lines:
        for word in line[5].split():
            if word not in all_noncs_words:
                all_noncs_words.append(word)
    zh_vocab = [word for word in all_noncs_words if hanzidentifier.is_simplified(word) and word != '、']

    vocab_dict = {}
    translator = Translator()
    for zh_word in zh_vocab:
        if len(translator.translate(zh_word).text.split()) == 1:
            vocab_dict[zh_word] = translator.translate(zh_word).text.lower()
    
    eng_word_pool = [word for word in set(vocab_dict.values())]

    # if we only want NN, include the following chunk:
    # nltk.download('brown')
    # nltk.download('averaged_perceptron_tagger')
    # corpus = brown.words()
    # brown_list = [word for (word, pos) in nltk.pos_tag(corpus) if pos.startswith(word_type)]
    # NN_pool = [word for word in eng_word_pool if word in brown_list]

    return eng_word_pool

def collect_10k_random_samples(eng_word_pool, word_list, word_vectors_array, kdtree):    
    means = []
    while len(means) < 10000:
        random_eng_words = random.sample(eng_word_pool, 220)
        results = calculate_vec_distance(word_list, word_vectors_array, kdtree, 'mixed', passed_in_words = random_eng_words)
        useful_results = [result for result in results if isinstance(result, tuple) and len(result) == 5]
        if len(useful_results) >= 199:
            sample = random.sample(useful_results, 199)
            for item in sample:
                tuple_to_write = item + (len(means)+1,)
                save_csv_a('/Users/yanting/Desktop/cs/results/random_noncs_words.csv', tuple_to_write)
            distances_str = [item[2] for item in sample]
            distances = [float(x) for x in distances_str]
            mean_distance = sum(distances)/len(distances)
            cos_sims_str = [item[4] for item in sample]
            cos_sims = [float(x) for x in cos_sims_str]  
            mean_cos_sim = sum(cos_sims)/len(cos_sims)
            result = (len(means)+1, mean_distance, mean_cos_sim)
            save_csv_a('/Users/yanting/Desktop/cs/results/results_random_noncs_words.csv', result)
            means.append(result)
            print(len(means))
        else:
            print('abandon sample')

    return means
 
def get_matching_noncs_words_for_cs_words():
    entire_file = []
    with open(extended_file_path, 'r', encoding = 'utf-8') as f:
        filereader = csv.reader(f, delimiter = ',')
        for row in filereader:
            entire_file.append(row)
            all_sents = entire_file[1:]

    matching_dict = {}
    n = 0
    while n < len(all_sents):
        if all_sents[n][30] == '1':
            cs_word = all_sents[n][7]
            cs_pos = all_sents[n][11]
            noncs_word = all_sents[n+1][8]

            translator = Translator()
            if len(translator.translate(noncs_word).text.split()) == 1:
                key = cs_word + '-' + cs_pos
                if key not in matching_dict.keys():
                    matching_dict[key] = [translator.translate(noncs_word).text.lower()]
                else:
                    matching_dict[key].append(translator.translate(noncs_word).text.lower())
        n += 2

    matching_NN_dict = {}
    for key, item in matching_dict.items():
        if key[-2:] == 'NN':
            matching_NN_dict[key] = item

    return matching_dict, matching_NN_dict

def collect_matching_noncswords_samples(matching_NN_dict, word_list, word_vectors_array, kdtree):    
    keys = matching_NN_dict.keys()
    cs_words = [item[:-3] for item in matching_NN_dict.keys()]
    matching_noncs_words = []
    for key in keys:
        if len(matching_NN_dict.get(key)) == 1 and key != matching_NN_dict.get(key[:-3]):
            matching_noncs_word = matching_NN_dict.get(key)[0]
        elif len(matching_NN_dict.get(key)) > 1:
            matching_pool = [word for word in matching_NN_dict.get(key) if word != key[:-3]]
            matching_noncs_word = random.sample(matching_pool, 1)[0]
        matching_noncs_words.append(matching_noncs_word)
    
    cs_results = calculate_vec_distance(word_list, word_vectors_array, kdtree, 'mixed', passed_in_words = cs_words)
    noncs_results = calculate_vec_distance(word_list, word_vectors_array, kdtree, 'mixed', passed_in_words = matching_noncs_words)
    
    n = 0
    while n < len(cs_results):
        if isinstance(cs_results[n], tuple) and len(cs_results[n]) == 5 and isinstance(noncs_results[n], tuple) and len(noncs_results[n]) == 5:
            combined_result = cs_results[n] + noncs_results[n]
            save_csv_a('/Users/yanting/Desktop/cs/results/results_matching_NN5.csv', combined_result)
        n += 1

def load_dict(dict_file):
    with open(dict_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    dictionary = {}
    for i in range(len(lines)):
        pair = lines[i].split()
        eng = pair[0]
        zh = pair[1]
        dictionary[eng] = zh
    return dictionary

def create_extended_corpus():
    all_lines = save_whole_file(original_file_path)[0]
    
    # add line number to each line
    n = 0
    while n < len(all_lines):
        all_lines[n].append(n+1)
        n += 1
    
    # add cs_word_length to each line (for nonCS line, it will just be 0)
    eng_pattern = re.compile(r'[a-zA-Z]+')
    n = 0
    while n < len(all_lines):
        cs_word_idx = int(all_lines[n][6])
        cs_sent = all_lines[n][4]
        i = 0
        while i < len(cs_sent.split()[cs_word_idx:]):
            if not eng_pattern.search(cs_sent.split()[cs_word_idx:][i]):
                break
            i += 1
        all_lines[n].append(i+1)
        n += 2
    m = 1
    while m < len(all_lines):
        all_lines[m].append(0)
        m += 2

    # for each cs sentence that has only 1 cs word, attach that word's Chinese translation (out of context) provided by chatGPT
    gpt_dict = load_dict("/Users/yanting/Desktop/cs/data/gpt_dict.txt")
    for line in all_lines:
        if line[30] == 1:
            trans = gpt_dict.get(line[7])
            print(line[7],trans)
        else:
            trans = ''
        line.append(trans)
    
    # for each non-cs sentence, attach the non-CS word's English translation (out of context) provided by chatGPT
    

if __name__ == "__main__":
    original_file_path = '/Users/yanting/Desktop/cs/data/original1476.csv'
    extended_file_path = '/Users/yanting/Desktop/cs/data/extended_gpt.csv'
    # eng_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/wiki.en.align.vec'
    eng_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/short_eng.vec'
    eng_kdtree_path = '/Users/yanting/Desktop/cs/word_vector/eng_kdtree.pkl'
    # zh_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/wiki.zh.align.vec'
    zh_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/short_zh.vec'
    zh_kdtree_path = '/Users/yanting/Desktop/cs/word_vector/zh_kdtree.pkl'
    shared_vec_path = '/Users/yanting/Desktop/cs/word_vector/shared_30k.vec'
    # test_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/testeng.vec'
    # test_kdtree_path = '/Users/yanting/Desktop/cs/word_vector/test_kdtree.pkl'
    

    # eng_words, trans_words, zh_words = word_collector(original_file_path)
    # eng_vec_dict = eng_vector_finder_easy(original_file_path, eng_word_vec_path)
    # trans_vec_dict = trans_vector_finder(original_file_path, zh_word_vec_path)
    # eng_wordlen_dict = eng_wordlen_calculator(original_file_path)
    # pinyinlen_dict, charlen_dict = trans_wordlen_calculator(original_file_path)
    # cos_sim_list = gather_cos_similarity(original_file_path, eng_word_vec_path, zh_word_vec_path)
    # save_cos_similarity('/Users/yanting/Desktop/cs/word_vector/cosine_similarity.csv', cos_sim_list)
    # eng_word_list, eng_word_vectors_array, eng_kdtree = create_KDtree(eng_word_vec_path)
    # zh_word_list, zh_word_vectors_array, zh_kdtree = create_KDtree(zh_word_vec_path)
    # shared_word_list, shared_word_vectors_array, shared_kdtree = create_KDtree(shared_vec_path)
    # save_kdtree(zh_kdtree, zh_kdtree_path)
    # eng_results = calculate_vec_distance(eng_word_list, eng_word_vectors_array, eng_kdtree, 'eng')
    # zh_results = calculate_vec_distance(zh_word_list, zh_word_vectors_array, zh_kdtree, 'zh')
    
    eng_word_pool = get_random_words_corpus()
    print('random_word_list_ready')
    shared_word_list, shared_word_vectors_array, shared_kdtree = create_KDtree(shared_vec_path)
    print('kdtree ready')
    means = collect_10k_random_samples(eng_word_pool,shared_word_list, shared_word_vectors_array, shared_kdtree)


# code-switch,PSU,380,PSU_2600,小区 环境 优美 安静 ， 物业 非常 nice 。,小区 环境 优美 安静 ， 物业 非常 友善 。,8,nice,友善,40.69440709423323,3.261233,VA,3,conj,5.445959 4.471926 6.066541 6.265982 0.8755206 4.950842 4.628391 3.261233 0.3388186,4.0339125777777785,7.792737443373911,12.012245148550017,4,2,5,0,0,4.628391,3,2,3,2,9
# non-code-switch,PSU,2600,PSU_380, ,房子 整洁 干净 ， 设施 很 好 。,7,,好,33.99205262349187,3.559633,VA,2,conj,5.512905 6.769275 2.024773 0.69897 4.819738 4.030354 3.559633 1.655329,3.633872125,,6.786498474836816,,1,5,0,0,4.030354,3,2,3,2,8

# the original file has the following info:
# 0. sent_type: code-switch
# 1. university: CMU
# 2. sent_id: 2
# 3. aligned_to: CMU_296
# 4. original_sentence: rich neighborhood ， 非常 安全 （ 这里 属于 富人 区 ） ， 旁边 就 是 公园 ， 附近 还有 配套 的 各 种 球场 和 社区 设施 ， 环境 很 好 很 安静 。
# 5. translation: 富人 区 ， 非常 安全 （ 这里 属于 富人 区 ） ， 旁边 就 是 公园 ， 附近 还有 配套 的 各 种 球场 和 社区 设施
# ， 环境 很 好 很 安静 。
# 6. word_id: 1
# 7. first_cs_word_form: rich
# 8. first_cs_word_translation: 富人
# 9. frequency_negative_ln_first_cs_word_trans: 46.615352726427304
# 10. surprisal_first_cs_word_trans: 6.100998
# 11. pos_tag_first_cs_word_trans: NN
# 12. governor_first_cs_word_trans: 2
# 13. deprel_first_cs_word_trans: compound:nn
# 14. surprisal_values: 6.100998 1.320391 0.7630151 4.4722 3.20577 2.76481 3.874677 3.395841 5.984178 1.320391 2.043565 0.7296352 4.6413 3.554084 2.235204 4.472754 1.254587 2.761207 1.42266 4.00532 0.3216881 3.688545 4.236028 5.883419 2.087959 3.743304 4.819367 0.6720446 4.722296 3.517969 3.595524 5.194436 3.258603 0.8566467
# 15. average_surprisal: 3.144718138235294
# 16. bilingual_corpus_frequency_negative_log_first_cs_word: 11.319097967990071
# 17. bilingual_corpus_frequency_negative_log_first_cs_word_trans: 10.402807236115917
# 18. length_first_cs_word_form: 4
# 19. length_first_cs_word_trans: 2
# 20. dependency_distance: 1
# 21. if_it_is_root: 0
# 22. if_previous_word_is_punctuation: (null)
# 23. surprisal_of_previous_word: (null)
# 24. 25_50_25_percent_location: 1
# 25. 10_80_10_percent_location: 1
# 26. 30_40_30_percent_location: 1
# 27. first_middle_last_location: 1
# 28. translation_sentence_length: 34