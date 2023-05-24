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

    translator = Translator()
    zh_words_trans = [translator.translate(word).text for word in zh_words]    
    zh_words_trans = [word.lower() for word in zh_words_trans]
    
    return eng_words, trans_words, zh_words, zh_words_trans

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
            line = (word, str(vec))
            f.writelines(line)

def calculate_cosine_similarity(eng, zh):
    cosine_similarity = np.dot(eng, zh) / (np.linalg.norm(eng) * np.linalg.norm(zh))
    return cosine_similarity

def gather_cos_similarity(original_file_path, eng_word_vec_path, zh_word_vec_path):
    cs_lines = save_whole_file(original_file_path)[1]
    eng_vec_dict = eng_vector_finder_easy(original_file_path, eng_word_vec_path)[1]
    zh_vec_dict = zh_vector_finder(original_file_path, zh_word_vec_path)
    cos_sim_list = []

    for line in cs_lines:
        eng_word = line[7]
        trans_word = line[8]
        sent_id = line[2]
        if eng_word in eng_vec_dict.keys() and trans_word in zh_vec_dict.keys():
            eng_wordvec = eng_vec_dict.get(eng_word)
            trans_wordvec = zh_vec_dict.get(trans_word)
            cos_sim = calculate_cosine_similarity(eng_wordvec, trans_wordvec)
            entry = (sent_id, eng_word, trans_word, cos_sim)
            cos_sim_list.append(entry)
    
    return cos_sim_list

def save_cos_similarity(output_file, cos_sim_list):
    with open(output_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerows(cos_sim_list)

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

if __name__ == "__main__":
    original_file_path = '/Users/yanting/Desktop/cs/data/original1476.csv'
    eng_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/wiki.en.align.vec'
    zh_word_vec_path = '/Users/yanting/Desktop/cs/word_vector/wiki.zh.align.vec'
    
    # eng_words, trans_words, zh_words = word_collector(original_file_path)
    # eng_vec_dict = eng_vector_finder_easy(original_file_path, eng_word_vec_path)
    # trans_vec_dict = trans_vector_finder(original_file_path, zh_word_vec_path)
    # eng_wordlen_dict = eng_wordlen_calculator(original_file_path)
    # pinyinlen_dict, charlen_dict = trans_wordlen_calculator(original_file_path)
    cos_sim_list = gather_cos_similarity(original_file_path, eng_word_vec_path, zh_word_vec_path)
    save_cos_similarity('/Users/yanting/Desktop/cs/word_vector/cosine_similarity.csv', cos_sim_list)

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