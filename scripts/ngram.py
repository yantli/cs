# trying to use kenlm for ngram trainning

# 5/4/2023

## create a new environment for ngram training:
# conda create --name ngramenv

## install kenlm (the following three chunks of code were what I tried but it seems not enough to get me to the ngram training):
# python -m pip install pypi-kenlm

# wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
# mkdir kenlm/build
# cd kenlm/build
# cmake ..
# make -j2

# git clone https://github.com/kpu/kenlm.git

## so maybe try from here:
## inside the /Users/yanting/Desktop/cs/ngram/ directory, run the following to install Homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

## and then use brew to install a bunch of things:
# brew install cmake
# brew install llvm
# brew install eigen
# brew install boost

## and use the following to install kenlm
# cd /Users/yanting/Desktop/cs/ngram/kenlm      
# mkdir build
# cd build
# cmake ..
# make -j4         
# sudo make install

## then inside of /Users/yanting/Desktop/cs/ngram/kenlm/build/bin/ we can run the ngram training by using this:
# ./lmplz -o 5 --text <tokenized_text_path> --arpa <ngram_path>
## where the <tokenized_text_path> is a .txt file and <ngram_path> is a .arpa file

## and to query the trained ngram, still inside of the same directory, run:
# ./build_binary <ngram_path> <readable_ngram_path>
## where the <readable_ngram_path> is a .binary file


import kenlm

