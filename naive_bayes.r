
library('tidyverse')
library('bnlearn')
library('here')

ptm <- proc.time()

file_dir <- here()

set_level <- function(x, n = 2){levels(x) <- as.character(0:(n-1)) 
        return(x)
}

file_path <- file.path(file_dir,"newest_all_tokenized_data.zip")

print('loading data')

imdb_train <- read.table(unz(file_path, "tokenized_bayes_train_from_imdb_file_.csv"), header=FALSE, sep=',')
imdb_train <- imdb_train %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  filter_at(vars(-class), any_vars(. != 0)) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

s140_train <- read.table(unz(file_path, "tokenized_bayes_train_from_s140_file_.csv"), header=FALSE, sep=',')
s140_train <- s140_train %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  filter_at(vars(-class), any_vars(. != 0)) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

n_imdb <- length(imdb_train$class)
n_s140 <- length(s140_train$class)

train <- list()

print('spliting data')

set.seed(11235813) 

a <- c(rep(0, length(imdb_train)-1), rep(1,1))
b <- rep(1, length(imdb_train))
c <- c(rep(1, length(imdb_train)-1), rep(0,1))
d <- rep(0, length(imdb_train))

train[[1]] <- rbind(imdb_train[1:ceiling(.2*n_imdb),],a,b,c,d)
train[[2]] <- rbind(imdb_train[ceiling(.2*n_imdb+1):ceiling(.4*n_imdb),],a,b,c,d)
train[[3]] <- rbind(imdb_train[ceiling(.4*n_imdb+1):ceiling(.6*n_imdb),],a,b,c,d)
train[[4]] <- rbind(s140_train[1:ceiling(.2*n_s140),],a,b,c,d)
train[[5]] <- rbind(s140_train[ceiling(.2*n_s140+1):ceiling(.4*n_s140),],a,b,c,d)
train[[6]] <- rbind(s140_train[ceiling(.4*n_s140+1):ceiling(.6*n_s140),],a,b,c,d)
train[[7]] <- rbind(imdb_train[ceiling(.6*n_imdb+1):ceiling(.7*n_imdb),], s140_train[ceiling(.6*n_s140+1):ceiling(.7*n_s140),],a,b,c,d)
train[[7]] <- train[[7]][sample(nrow(train[[7]])),]
train[[8]] <- rbind(imdb_train[ceiling(.7*n_imdb+1):ceiling(.8*n_imdb),], s140_train[ceiling(.7*n_s140+1):ceiling(.8*n_s140),],a,b,c,d)
train[[8]] <- train[[8]][sample(nrow(train[[8]])),]
train[[9]] <- rbind(imdb_train[ceiling(.8*n_imdb+1):ceiling(.9*n_imdb),], s140_train[ceiling(.8*n_s140+1):ceiling(.9*n_s140),],a,b,c,d)
train[[9]] <- train[[9]][sample(nrow(train[[9]])),]
train[[10]] <- rbind(imdb_train[ceiling(.9*n_imdb+1):n_imdb,], s140_train[ceiling(.9*n_s140+1):n_s140,],a,b,c,d)
train[[10]] <- train[[10]][sample(nrow(train[[10]])),]

print('loading test data')

amazon <- read.table(unz(file_path, "tokenized_test_from_amazon_file_.csv"), header=FALSE, sep=',')
amazon_df <- enframe(amazon$V986, name=NULL) %>% rename(id = value)
amazon <- amazon %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

yelp <- read.table(unz(file_path, "tokenized_test_from_yelp_file_.csv"), header=FALSE, sep=',')
yelp_df <- enframe(yelp$V986, name=NULL) %>% rename(id = value)
yelp <- yelp %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

roberta_imdb <- read.table(unz(file_path, "tokenized_roberta_train_from_imdb_file_.csv"), header=FALSE, sep=',')
roberta_imdb_df <- enframe(roberta_imdb$V986, name=NULL) %>% rename(id = value)
roberta_imdb <- roberta_imdb %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

roberta_s140 <- read.table(unz(file_path, "tokenized_roberta_train_from_s140_file_.csv"), header=FALSE, sep=',')
roberta_s140_df <- enframe(roberta_s140$V986, name=NULL) %>% rename(id = value)
roberta_s140 <- roberta_s140 %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

imdb_test <- read.table(unz(file_path, "tokenized_test_from_imdb_file_.csv"), header=FALSE, sep=',')
imdb_test_df <- enframe(imdb_test$V986, name=NULL) %>% rename(id = value)
imdb_test <- imdb_test %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)

s140_test <- read.table(unz(file_path, "tokenized_test_from_s140_file_.csv"), header=FALSE, sep=',')
s140_test_df <- enframe(s140_test$V986, name=NULL) %>% rename(id = value)
s140_test <- s140_test %>% 
  mutate_all(sign) %>%
  select(-V986) %>%
  rename(class = V985) %>%
  mutate_all(as.factor) %>%
  mutate_all(set_level)
  
print('training models')

data_dir <- file.path(file_dir,"data","naive")

training_time <- proc.time() - proc.time()
testing_time <- proc.time() - proc.time()

amazon_acc <- 0
yelp_acc <- 0
roberta_imdb_acc <- 0
roberta_s140_acc <- 0
imdb_test_acc <- 0
s140_test_acc <- 0



for(i in 1:10){
  ptm <- proc.time()
  
  nb <- naive.bayes(train[[i]], "class")
  fitted <- bn.fit(nb, train[[i]])
  
  training_time <- training_time + (proc.time() - ptm)
  ptm <- proc.time()
  
  pred <- predict(fitted, amazon, prob=TRUE)
  amazon_acc <- amazon_acc + sum(pred[1:length(amazon$class)] == amazon$class)/length(amazon$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  amazon_df <- add_column(amazon_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  
  pred <- predict(fitted, yelp, prob=TRUE)
  yelp_acc <- yelp_acc + sum(pred[1:length(yelp$class)] == yelp$class)/length(yelp$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  yelp_df <- add_column(yelp_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  pred <- predict(fitted, roberta_imdb, prob=TRUE)
  roberta_imdb_acc <- roberta_imdb_acc + sum(pred[1:length(roberta_imdb$class)] == roberta_imdb$class)/length(roberta_imdb$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  roberta_imdb_df <- add_column(roberta_imdb_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  pred <- predict(fitted, roberta_s140, prob=TRUE)
  roberta_s140_acc <- roberta_s140_acc + sum(pred[1:length(roberta_s140$class)] == roberta_s140$class)/length(roberta_s140$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  roberta_s140_df <- add_column(roberta_s140_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  pred <- predict(fitted, imdb_test, prob=TRUE)
  imdb_test_acc <- imdb_test_acc + sum(pred[1:length(imdb_test$class)] == imdb_test$class)/length(imdb_test$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  imdb_test_df <- add_column(imdb_test_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  pred <- predict(fitted, s140_test, prob=TRUE)
  s140_test_acc <- s140_test_acc + sum(pred[1:length(s140_test$class)] == s140_test$class)/length(s140_test$class)
  prob <- attr(pred,'prob')
  for(j in 1:length(pred)){
    if(is.nan(prob[1,j])){
      prob[1,j] <- (1-as.numeric(as.character(pred[j])))*.49 +.02
      prob[2,j] <- as.numeric(as.character(pred[j]))*.49 +.02
    }
  }
  prob <- as.data.frame(t(prob))
  s140_test_df <- add_column(s140_test_df, !!as.character(i):=prob$`1`)
  rm(pred)
  rm(prob)
  
  print(i)
  testing_time <- testing_time + (proc.time() - ptm)
}

amazon_acc <- amazon_acc/i
yelp_acc <- yelp_acc/i
roberta_imdb_acc <- roberta_imdb_acc/i
roberta_s140_acc <- roberta_s140_acc/i
imdb_test_acc <- imdb_test_acc/i
s140_test_acc <- s140_test_acc/i

print("amazon_acc:")
print(amazon_acc)
print("yelp_acc:")
print(yelp_acc)
print("roberta_imdb_acc:")
print(roberta_imdb_acc)
print("roberta_s140_acc:")
print(roberta_s140_acc)
print("imdb_test_acc:")
print(imdb_test_acc)
print("s140_test_acc:")
print(s140_test_acc)

write.csv(amazon_df,file.path(data_dir,paste('amazon_prob_naive.csv',sep='')))
write.csv(yelp_df,file.path(data_dir,paste('yelp_prob_naive.csv',sep='')))
write.csv(roberta_imdb_df,file.path(data_dir,paste('roberta_imdb_prob_naive.csv',sep='')))
write.csv(roberta_s140_df,file.path(data_dir,paste('roberta_s140_prob_naive.csv',sep='')))
write.csv(imdb_test_df,file.path(data_dir,paste('imdb_test_prob_naive.csv',sep='')))
write.csv(s140_test_df,file.path(data_dir,paste('s140_test_prob_naive.csv',sep='')))

print('training time:')
print(training_time)

print('testing time:')
print(testing_time)
