library('tidyverse')
library('bnlearn')
library('here')

args = commandArgs(trailingOnly=TRUE)

file_dir <- here()
if(length(args) > 0){
  train_path <- file.path(file_dir,args[1])
} else {
  train_path <- file.path(file_dir,"tokenized_train_imdb.csv")
}
if(length(args) > 1){
  test_path <- file.path(file_dir,args[2])
} else {
  test_path <- file.path(file_dir,"tokenized_test_imdb.csv")
}

train <- read.table(train_path, header=TRUE, sep=',')
train <- mutate_all(train, sign)
train <- select(train, -X)
train <- train[sample(nrow(train)),]
train[colnames(train)] <- lapply(train[colnames(train)], factor)

test <- read.table(test_path, header=TRUE, sep=',')
test <- mutate_all(test, sign)
test <- select(test, -X)
test <- test[sample(nrow(test)),]
test[colnames(test)] <- lapply(test[colnames(test)], factor)

nb <- naive.bayes(train, "class")

fitted <- bn.fit(nb, train)

pred <- predict(fitted, test, prob=TRUE)

n <- length(pred)

sum(test[,'class'] == pred[1:n])/n

