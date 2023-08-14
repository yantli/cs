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
library(boot)

load_matching_data <- function(file) {
  data <- read_delim(file, locale=locale(encoding="UTF-8"),
                     col_names = c("cs_word",
                                   "cs_closest_zh_neighbor",
                                   "cs_distance",
                                   "cs_nth_closest",
                                   "cs_cos_sim",
                                   "matching_noncs_word",
                                   "noncs_closest_zh_neighbor",
                                   "noncs_distance",
                                   "noncs_nth_closest",
                                   "noncs_cos_sim"))
  return(data)
}

matching <- load_matching_data("/Users/yanting/Desktop/cs/results/results_matching_NN1.csv")
matching$cs_distance <- as.numeric(matching$cs_distance)
matching$cs_cos_sim <- as.numeric(matching$cs_cos_sim)
matching$noncs_distance <- as.numeric(matching$noncs_distance)
matching$noncs_cos_sim <- as.numeric(matching$noncs_cos_sim)
summary(matching$cs_distance)
summary(matching$noncs_distance)
summary(matching$cs_cos_sim)
summary(matching$noncs_cos_sim)
t.test(matching$cs_distance, matching$noncs_distance, paired = TRUE)
t.test(matching$cs_cos_sim, matching$noncs_cos_sim, paired = TRUE)



load_dis_cossim_data <- function(file) {
  data <- read_delim(file, locale=locale(encoding="UTF-8"),
                     col_names = c("sample_number",
                                   "mean_distance",
                                   "mean_cos_sim"))
  return(data)
}

path_to_random500 <- "/Users/yanting/Desktop/cs/results/results_random_500.csv"
path_to_random10k <- "/Users/yanting/Desktop/cs/results/results_random_noncs_words.csv"
dis_cossim <- load_dis_cossim_data(path_to_random10k)
summary(dis_cossim$mean_distance)
sd(dis_cossim$mean_distance)
summary(dis_cossim$mean_cos_sim)

boot_distance <- boot(dis_cossim$mean_distance,function(u,i) mean(u[i]),R=1000)
boot_dis_result <- boot.ci(boot_distance,type=c("norm","basic","perc"))
lapply(boot_dis_result, function(ci) format(ci, digits = 10))

boot_cos_sim <- boot(dis_cossim$mean_cos_sim,function(u,i) mean(u[i]),R=1000)
boot_cos_result <- boot.ci(boot_cos_sim,type=c("norm","basic","perc"))
lapply(boot_cos_result, function(ci) format(ci, digits = 10))


ggplot(data.frame(x = dis_cossim$mean_distance), aes(x)) +
  geom_density(color = "blue") +
  geom_point(data = data.frame(x = 1.0580), aes(x, y = 0.01), color = "red", size = 3) +
  labs(title = "Density Plot of Distance")
ggplot(data.frame(x = dis_cossim$mean_cos_sim), aes(x)) +
  geom_density(color = "purple") +
  geom_point(data = data.frame(x = 0.4386), aes(x, y = 0.01), color = "red", size = 3) +
  labs(title = "Density Plot of Cos_sim")




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

noncsNN1 <- load_data("/Users/yanting/Desktop/cs/word_vector/randomNN_closest_zhneighbor_simp.csv")
noncsNN1$distance <- as.numeric(noncsNN1$distance)
noncsNN1$cos_sim <- as.numeric(noncsNN1$cos_sim)

noncsNN2 <- load_data("/Users/yanting/Desktop/cs/word_vector/randomNN_closest_zhneighbor_simp2.csv")
noncsNN2$distance <- as.numeric(noncsNN2$distance)
noncsNN2$cos_sim <- as.numeric(noncsNN2$cos_sim)

t.test(csdata$distance, noncs1$distance, var.equal = FALSE)
t.test(csdata$distance, noncs2$distance, var.equal = FALSE)
t.test(csdata$distance, noncs3$distance, var.equal = FALSE)
t.test(csdata$distance, noncsNN1$distance, var.equal = FALSE)
t.test(csdata$distance, noncsNN2$distance, var.equal = FALSE)

t.test(csdata$cos_sim, noncs1$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncs2$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncs3$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncsNN1$cos_sim, var.equal = FALSE)
t.test(csdata$cos_sim, noncsNN2$cos_sim, var.equal = FALSE)

