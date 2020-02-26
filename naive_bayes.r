library('tidyverse')
library('bnlearn')

alm <- alarm

nb <- naive.bayes(alm, "CVP")



y <- head(alm[2:37], 20)

fitted <- bn.fit(nb, alm)

pred <- predict(fitted, y, prob=TRUE)

table(pred,alm[1][1:20,])
