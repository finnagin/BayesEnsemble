library('tidyverse')
library('bnlearn')
library('here')

args = commandArgs(trailingOnly=TRUE)

args

file_dir <- here()
csv_path <- file.path(file_dir,"tokenized_train_imdb.csv")

df <- read.table(csv_path, header=TRUE, sep=',')
df <- mutate_all(df, sign)
df <- select(df, -X)
df <- df[sample(nrow(df)),]
df[colnames(df)] <- lapply(df[colnames(df)], factor)


train <- head(df,18000)

test <- tail(df,1261)

nb <- naive.bayes(train, "class")

fitted <- bn.fit(nb, train)

pred <- predict(fitted, test, prob=TRUE)

n <- length(pred)

sum(test[,'class'] == pred[1:n])/n
