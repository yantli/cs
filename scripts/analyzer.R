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
library(patchwork)
library(gridExtra)


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

matching <- load_matching_data("/Users/yanting/Desktop/cs/results/results_matching_NN6.csv")
matching$cs_distance <- as.numeric(matching$cs_distance)
matching$cs_cos_sim <- as.numeric(matching$cs_cos_sim)
matching$noncs_distance <- as.numeric(matching$noncs_distance)
matching$noncs_cos_sim <- as.numeric(matching$noncs_cos_sim)
mean(matching$cs_distance)
mean(matching$noncs_distance)
summary(matching$cs_cos_sim)
summary(matching$noncs_cos_sim)
t.test(matching$cs_distance, matching$noncs_distance, paired = TRUE)
t.test(matching$cs_cos_sim, matching$noncs_cos_sim, paired = TRUE)

matching_without_outliers <- matching %>%
  filter(!(cs_word %in% c('ps', 'apt', 'topic', 'acc', 'link', 'thread', 'coverage', 'gre', 'ta', 'ie', 'firestone')))
mean(matching_without_outliers$cs_distance)
mean(matching_without_outliers$cs_cos_sim)
t.test(matching_without_outliers$cs_distance, matching_without_outliers$noncs_distance, paired = TRUE)
t.test(matching_without_outliers$cs_cos_sim, matching_without_outliers$noncs_cos_sim, paired = TRUE)

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

csdata <- load_data("/Users/yanting/Desktop/cs/results/csNN_closest_zhneighbor_simp.csv")
csdata$distance <- as.numeric(csdata$distance)
csdata$cos_sim <- as.numeric(csdata$cos_sim)
mean(csdata$distance)
mean(csdata$cos_sim)

noncsdata <- load_data("/Users/yanting/Desktop/cs/results/random_noncs_words_uniq1425.csv")
noncsdata$distance <- as.numeric(noncsdata$distance)
noncsdata$cos_sim <- as.numeric(noncsdata$cos_sim)
mean(noncsdata$distance)
mean(noncsdata$cos_sim)

combined_data <- bind_rows(csdata %>% mutate(Source = "cs"),
                           noncsdata %>% mutate(Source = "noncs"))
ggplot(combined_data, aes(x = Source, y = distance)) +
  geom_boxplot() +
  labs(x = "Source", y = "distance") 
ggplot(combined_data, aes(x = Source, y = cos_sim)) +
  geom_boxplot() +
  labs(x = "Source", y = "cos_sim")


# bootstrapping based on the 1425 uniq non cs words:
boot_distance <- boot(noncsdata$distance,function(u,i) mean(u[i]),R=1000)
boot_dis_result <- boot.ci(boot_distance,type=c("norm","basic","perc"))
lapply(boot_dis_result, function(ci) format(ci, digits = 10))

# Extract bootstrapped values
boot_dis_values <- boot_distance$t

# Create a density plot of bootstrapped values
ggplot(data = data.frame(Values = boot_dis_values), aes(x = Values)) +
  geom_density(fill = "blue", alpha = 0.5) +
  geom_vline(xintercept = boot_dis_result$normal[4], color = "red", linetype = "dashed") +
  geom_vline(xintercept = boot_dis_result$basic[4], color = "blue", linetype = "dashed") +
  geom_vline(xintercept = boot_dis_result$percent[4], color = "green", linetype = "dashed") +
  labs(x = "Bootstrapped Values", y = "Density") +
  theme_minimal()
p1<-ggplot(data.frame(Values = boot_dis_values), aes(x = Values)) +
  geom_density(color = "blue") +
  geom_point(data = data.frame(x = 1.0626897), aes(x, y = 0.01), color = "red", size = 5) +
  labs(y = "Density", x = 'Distance', title = "Density plot of bootstrapped mean distance") +
  theme_bw() + theme(axis.text.x=element_text(size=12), axis.title=element_text(size=12), axis.text.y=element_text(size=12))


boot_cos_sim <- boot(noncsdata$cos_sim,function(u,i) mean(u[i]),R=1000)
boot_dis_result <- boot.ci(boot_cos_sim,type=c("norm","basic","perc"))
lapply(boot_dis_result, function(ci) format(ci, digits = 10))

boot_cos_values <- boot_cos_sim$t
p2<-ggplot(data.frame(Values = boot_cos_values), aes(x = Values)) +
  geom_density(color = "purple") +
  geom_point(data = data.frame(x = 0.4334456744), aes(x, y = 0.01), color = "red", size = 5) +
  labs(y = "Density", x = 'Cosine similarity', title = "Density plot of bootstrapped mean cosine similarity") +
  theme_bw() + theme(axis.text.x=element_text(size=12), axis.title=element_text(size=12), axis.text.y=element_text(size=12))

p <- grid.arrange(p1, p2, ncol=2)
ggsave("/Users/yanting/Desktop/cs/results/bootstrap.png", plot=p, height = 3, width = 10)



ggplot(data.frame(x = noncsdata$distance), aes(x)) +
  geom_density(color = "blue") +
  geom_point(data = data.frame(x = 1.0580), aes(x, y = 0.01), color = "red", size = 3) +
  labs(title = "Density Plot of Distance")
ggplot(data.frame(x = noncsdata$cos_sim), aes(x)) +
  geom_density(color = "purple") +
  geom_point(data = data.frame(x = 0.4386), aes(x, y = 0.01), color = "red", size = 3) +
  labs(title = "Density Plot of Cosine Similarity")




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

