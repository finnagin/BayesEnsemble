library('tidyverse')
library('bnlearn')
library('here')

args = commandArgs(trailingOnly=TRUE)

file_dir <- here()
if(length(args) > 0){
  train_name <- args[1]
  train_path <- file.path(file_dir,"tokenized_train_data.zip")
} else {
  train_name <- "tokenized_train_from_imdb_and_s140_file_1.csv"
  train_path <- file.path(file_dir,"tokenized_train_data.zip")
}
if(length(args) > 1){
  test_name <- args[2]
  test_path <- file.path(file_dir,"tokenized_train_data.zip")
} else {
  test_name <- "tokenized_train_from_imdb_and_s140_file_2.csv"
  test_path <- file.path(file_dir,"tokenized_train_data.zip")
}

set_level <- function(x, n = 2){levels(x) <- as.character(0:(n-1)) 
        return(x)
        }

train <- read.table(unz(train_path, train_name), header=TRUE, sep=',')
train <- mutate_all(train, sign)
train <- select(train, -X)
train <- mutate_all(train, as.factor)
train <- mutate_all(train, set_level)

test <- read.table(unz(test_path, test_name), header=TRUE, sep=',')
test <- mutate_all(test, sign)
test_class <- test$class
test <- select(test, -X, -class)
test <- mutate_all(test, as.factor)
test <- mutate_all(test, set_level)

nb <- naive.bayes(train, "class")

fitted <- bn.fit(nb, train)

pred <- predict(fitted, test, prob=TRUE)

n <- length(pred)

#   sum(test_class == pred[1:n])/n

if(length(args) > 2){
  #pred_path <- file.path(file_dir,paste("preds_",args[3],".csv",sep=""))
  prob_path <- file.path(file_dir,paste("probs_",args[3],".csv",sep=""))
  #write.csv(pred[1:n],pred_path)
  write.csv(attr(pred,'prob'),prob_path)
}
