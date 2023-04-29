setwd("~/Desktop/dottorato/neural_networks_phd_project")
library(ggplot2)
library(tidyverse)

path = "results/classification_twoLNN_1_0.05_128_20.csv"
df = read.csv(path, sep = ",")

df_long <- df %>%
  select(-X) %>%
  pivot_longer(!epoch, values_to = "accuracy") %>%
  mutate(name = factor(name, levels = c("train_acc", "test_acc"))) %>%
  rename(dataset = name)

ggplot(df_long, mapping = aes(x=epoch, y=accuracy, fill=dataset, color=dataset)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  lims(y=c(.4,1)) +
  scale_color_manual(values = c("steelblue", "darkorange"), labels = c("Test", "Train")) +
  theme(legend.position = "bottom")


