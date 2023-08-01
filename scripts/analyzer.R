# to analyze the vector distance and cosine similarity
# Yanting Li
# 7/31/2023

library(tidyverse)
library(dplyr)
library(showtext)
showtext_auto()
library(lme4)
library(lmerTest)
library(ggplot2)
library(poolr)
library(ggrepel)


load_data <- function(file) {
  data <- read_delim(file, locale=locale(encoding="UTF-8"),
                     col_names = c("eng_word",
                                   "closest_zh_neighbor",
                                   "distance",
                                   "nth_closest",
                                   "cos_sim"))
  return(data)
}

csdata <- load_data("/Users/yanting/Desktop/cs/word_vector/cswords_closest_zhneighbor_simp.csv")
csdata$distance <- as.numeric(csdata$distance)
csdata$cos_sim <- as.numeric(csdata$cos_sim)

noncs1 <- load_data("/Users/yanting/Desktop/cs/word_vector/randomwords_closest_zhneighbor_simp1.csv")
noncs1$distance <- as.numeric(noncs1$distance)
noncs1$cos_sim <- as.numeric(noncs1$cos_sim)

noncs2 <- load_data("/Users/yanting/Desktop/cs/word_vector/randomwords_closest_zhneighbor_simp2.csv")
noncs2$distance <- as.numeric(noncs2$distance)
noncs2$cos_sim <- as.numeric(noncs2$cos_sim)

noncs3 <- load_data("/Users/yanting/Desktop/cs/word_vector/randomwords_closest_zhneighbor_simp3.csv")
noncs3$distance <- as.numeric(noncs3$distance)
noncs3$cos_sim <- as.numeric(noncs3$cos_sim)

t.test(csdata$distance, noncs1$distance, var.equal = FALSE)
t.test(csdata$distance, noncs2$distance, var.equal = FALSE)
t.test(csdata$distance, noncs3$distance, var.equal = FALSE)

t.test(csdata$cos_sim, noncs1$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncs2$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncs3$cos_sim, var.equal = FALSE)
