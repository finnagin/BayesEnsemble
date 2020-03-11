library('tidyverse')
library('glm2')
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
train[colnames(train)] <- lapply(train[colnames(train)], factor)

test <- read.table(test_path, header=TRUE, sep=',')
test <- mutate_all(test, sign)
test_class <- test$class
test <- select(test, -X, -class)
test[colnames(test)] <- lapply(test[colnames(test)], factor)

fitted <- glm(class~., data = train, family = binomial())

pred <- predict(fitted, test, type='response') #Returns probabilities

n <- length(pred)

#sum(test_class == pred[1:n])/n

if(length(args) > 2){
  pred_path <- file.path(file_dir,paste("preds_",args[3],".csv",sep=""))
  prob_path <- file.path(file_dir,paste("probs_",args[3],".csv",sep=""))
  write.csv(pred[1:n],pred_path)
  write.csv(attr(pred,'prob'),prob_path)
}