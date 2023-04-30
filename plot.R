setwd("~/Desktop/dottorato/neural_networks_phd_project")
library(ggplot2)
library(tidyverse)

# Initial Plots

for (k in c(1,3)) {
  path = paste0("results/parity_twoLNN_", k, "_0.05_128_20.csv")
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
    lims(y=c(0.5,1)) +
    scale_color_manual(values = c("steelblue", "darkorange"), labels = c("Test", "Train")) +
    scale_fill_manual(values = c("steelblue", "darkorange"), labels = c("Test", "Train")) +
    ggtitle(paste0("Parity task, twoLNN, k = ", k)) +
    theme(legend.position = "bottom")
  ggsave(paste0("plots/parity_twoLNN_k", k, ".png"), dpi=300, width = 7, height = 5)
}



# Parity task comparison

res = dplyr::tibble()
for (k in 1:5) {
  for (model in c("twoLNN", "fiveLNN", "LeNet")) {
    path = paste0("results/parity_", model, "_", k, "_0.05_128_20.csv")
    df = read.csv(path, sep = ",")
    acc <- df %>% filter(epoch == max(epoch)) %>% pull(test_acc)
    res <- dplyr::bind_rows(res, dplyr::tibble(k=k, model=model, accuracy=acc))

  }
}

res %>%
  mutate(model = factor(model, levels = c("twoLNN", "fiveLNN", "LeNet"))) %>%
  ggplot(mapping = aes(x=k, y=accuracy, color=model, fill=model)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  lims(y=c(0,1)) +
  theme(legend.position = "bottom") +
  ggtitle("Comparison parity task")
ggsave("plots/comparison_parity_task.png", dpi=300, width = 6, height = 5)

# Parity vs. Transfer classification

res = dplyr::tibble()
for (k in c(1,3)) {
  for (model in c("twoLNN", "fiveLNN", "LeNet")) {
    for (task in c("parity", "classification")) {
      if (task == "parity") path = paste0("results/", task, "_", model, "_", k, "_0.05_128_20.csv")
      if (task == "classification") path = paste0("results/", task, "_", model, "_", k, "_0.01_128_20.csv")
      df = read.csv(path, sep = ",")
      acc <- df %>% filter(epoch == max(epoch)) %>% pull(test_acc)
      res <- dplyr::bind_rows(res, dplyr::tibble(k=k, model=model, accuracy=acc, task=task))
    }
  }
}

for (k in c(1,3)) {
  k_val = k
  res %>%
    mutate(model = factor(model, levels = c("twoLNN", "fiveLNN", "LeNet"))) %>%
    mutate(task = factor(task, levels = c("parity", "classification"))) %>%
    filter(k == k_val) %>%
    ggplot(mapping = aes(x=model, fill=task, y=accuracy)) +
    geom_bar(position = "dodge", stat = "identity") +
    scale_fill_manual(values = c("maroon", "seagreen"), labels = c("parity", "transfer classification")) +
    theme_bw() +
    lims(y=c(0,1)) +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Transfer learning  K=", k))
  ggsave(paste0("plots/transfer_learning_k", k, ".png"), dpi = 300, width = 6, height = 5)
}

# From scratch

res = dplyr::tibble()
for (k in c(1,3)) {
  for (model in c("twoLNN", "fiveLNN", "LeNet")) {
    for (task in c("parity", "classification", "scratch_classification")) {
      if (task == "parity") path = paste0("results/", task, "_", model, "_", k, "_0.05_128_20.csv")
      if (task == "classification") path = paste0("results/", task, "_", model, "_", k, "_0.01_128_20.csv")
      if (task == "scratch_classification") path = paste0("results/from-scratch-classification", "_", model, "_", k, "_0.001_128_20.csv")
      df = read.csv(path, sep = ",")
      acc <- df %>% filter(epoch == max(epoch)) %>% pull(test_acc)
      res <- dplyr::bind_rows(res, dplyr::tibble(k=k, model=model, accuracy=acc, task=task))
    }
  }
}

for (k_val in c(1,3)) {
  res %>%
    mutate(model = factor(model, levels = c("twoLNN", "fiveLNN", "LeNet"))) %>%
    mutate(task = factor(task, levels = c("parity", "classification", "scratch_classification"))) %>%
    filter(k == k_val) %>%
    ggplot(mapping = aes(x=model, fill=task, y=accuracy)) +
    geom_bar(position = "dodge", stat = "identity") +
    scale_fill_manual(values = c("maroon", "seagreen", "darkgoldenrod2"), labels = c("parity", "transfer classification", "classification")) +
    theme_bw() +
    lims(y=c(0,1)) +
    theme(legend.position = "bottom") +
    ggtitle(paste0("all task  K=", k_val))

  ggsave(paste0("plots/all_tasks_k", k_val, ".png"), dpi = 300, width = 6, height = 5)
}


