## clear environment, import packages, set seed for repeatability, set working directory and import .txt file directly from website.

rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")
setwd("~/Documents/r_projects/a1")

library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(DataExplorer)
library(doParallel)

set.seed(123)

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
rawData <- read.csv(file, header = FALSE)

## name dataframe columns, change "1" and "0" to "true" and False and chance the "Auth" column to a factor
names(rawData) <- c("Variance", "Skewness","Kurtosis", "Entropy", "Auth")
rawData <- rawData %>% mutate(Auth = ifelse(Auth == 1, "true", "false"))
rawData$Auth <- as.factor(rawData$Auth)

## exploratory visualization of the dataset
str(rawData)
introduce(rawData)
head(rawData)
summary(rawData)
plot_histogram(rawData)
plot_bar(rawData)
plot_qq(rawData)
plot_correlation(rawData[ ,1:4])

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index_10 <- createDataPartition(rawData$Auth, p=0.8, list = FALSE, times = 10) # returns numerical vector of the index of the observations to be included in the training set, repeat so we have 10 different test training sets for later
train_index <- train_index_10[ ,1] # tuining hyperparaemters over a single test set is suitable so split out first index
predictors <- names(rawData[-5]) # return vector of column names, removing "Auth" as it is the response variable

testData <- rawData[-train_index, ]

trainingPredictors <- rawData[train_index,predictors]
trainingResponse <- as.factor(rawData[train_index, "Auth"])



## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

## train and tune Naive Bayes classifier using caret
start_time <- Sys.time()
train_control <- trainControl(method = "cv", number = 10) # use 10 fold cross validation on the training set to asses model hyper parameters
tune_params <- expand.grid(usekernel = c(TRUE, FALSE), fL = 1:5, adjust = 1:5) # create tuning grid over available hyper parameters for NB model

bank_nb_mod1 <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params) # train model

plot(bank_nb_mod1) # plot accuracy across tuining grid

bank_nb_mod1$finalModel$tuneValue # return hyperparameters of "best" model

bank_nb_final <- bank_nb_mod1$finalModel # save the most acurate model


## train and tune lda clasifier using caret
tune_params <- expand.grid(dimen = 0:10) # create tuning grid over available hyper parameters for NB model

bank_lda_mod1 <- train(x = trainingPredictors, y = trainingResponse, method = "lda2", trControl = train_control, tuneGrid = tune_params)
plot(bank_lda_mod1)

bank_lda_mod1$finalModel$tuneValue
bank_lda_final <- bank_lda_mod1$finalModel

## Use clasifiers to predict class on test data set
pred_nb <- predict(bank_nb_final, newdata = testData)
conf_nb <- confusionMatrix(pred_nb$class, testData$Auth)

pred_lda <- predict(bank_lda_mod1, newdata = testData)
conf_lda <- confusionMatrix(pred_lda, testData$Auth)

## compare models over 10 differnt test/training data sets

train_control <- trainControl(method = "cv", number = 10)
tune_params_nb <- expand.grid(usekernel = bank_nb_final$tuneValue[1,2], fL = 1, adjust = bank_nb_final$tuneValue[1,3])
tune_params_lda <- expand.grid(dimen = bank_lda_final$tuneValue[1,1])

all_nb <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- rawData[train_index_10[ ,i],predictors]
  trainingResponse <- rawData[train_index_10[ ,i],"Auth"]
  testData <- rawData[-train_index_10[,i], ]
  
  bank_mod4ass_nb <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params_nb)
  
  pred <- predict(bank_mod4ass_nb, newdata = testData)
  
  confusionMatrix(pred, testData$Auth)

})

all_lda <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- rawData[train_index_10[ ,i],predictors]
  trainingResponse <- rawData[train_index_10[ ,i],"Auth"]
  testData <- rawData[-train_index_10[,i], ]
 
  bank_mod4ass_lda <- train(x = trainingPredictors, y = trainingResponse, method = "lda2", trControl = train_control, tuneGrid = tune_params_lda)
  
  pred <- predict(bank_mod4ass_lda, newdata = testData)
  
  confusionMatrix(pred, testData$Auth)
  
})

end_time <- Sys.time()
(runtime <- end_time - start_time)


## close cluster
stopCluster(cl)

lda_acuracy <- lapply(seq_len(length(all_lda)), function(i){
  unlist(all_lda[[i]]$overall["Accuracy"])
})

lda_Kappa <- lapply(seq_len(length(all_lda)), function(i){
  unlist(all_lda[[i]]$overall["Kappa"])
})

lda_Sens <- lapply(seq_len(length(all_lda)), function(i){
  unlist(all_lda[[i]]$byClass["Sensitivity"])
})

lda_Spec <- lapply(seq_len(length(all_lda)), function(i){
  unlist(all_lda[[i]]$byClass["Specificity"])
})

nb_acuracy <- lapply(seq_len(length(all_nb)), function(i){
  unlist(all_nb[[i]]$overall["Accuracy"])
})

nb_Kappa <- lapply(seq_len(length(all_nb)), function(i){
  unlist(all_nb[[i]]$overall["Kappa"])
})

nb_Sens <- lapply(seq_len(length(all_nb)), function(i){
  unlist(all_nb[[i]]$byClass["Sensitivity"])
})

nb_Spec <- lapply(seq_len(length(all_nb)), function(i){
  unlist(all_nb[[i]]$byClass["Specificity"])
})

perf_nb <- data.frame(unlist(nb_acuracy), unlist(nb_Kappa), unlist(nb_Sens), unlist(nb_Spec))

perf_lda <- data.frame(unlist(lda_acuracy), unlist(lda_Kappa), unlist(lda_Sens), unlist(lda_Spec))


