## clear environment, import packages, set seed for repeatability, set working directory and import .txt file directly from website.

rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")
setwd("~/Documents/r_projects/a1")

library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(DataExplorer)
library(doParallel)
library(esquisse)

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

modelData <- rawData

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index_10 <- createDataPartition(modelData$Auth, p=0.8, list = FALSE, times = 10) # returns numerical vector of the index of the observations to be included in the training set, repeat so we have 10 different test training sets for later
train_index <- train_index_10[ ,1] # tuining hyperparaemters over a single test set is suitable so split out first index
predictors <- names(modelData[-5]) # return vector of column names, removing "Auth" as it is the response variable

testData <- modelData[-train_index, ]

trainingPredictors <- modelData[train_index,predictors]
trainingResponse <- as.factor(modelData[train_index, "Auth"])

rm(rawData, file)
## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

## train and tune Naive Bayes classifier using caret
start_time <- Sys.time()

train_control <- trainControl(method = "cv", number = 10) # use 10 fold cross validation on the training set to asses model hyper parameters
tune_params <- expand.grid(usekernel = c(TRUE, FALSE), fL = 1:5, adjust = 1:5) # create tuning grid over available hyper parameters for NB model

bank_nb_tune <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params, metric = "Accuracy") # train model

plot(bank_nb_tune) # plot accuracy across tuining grid

bank_nb_tune$finalModel$tuneValue # return hyperparameters of "best" model

bank_nb_finalHP <- bank_nb_tune$finalModel # save the most acurate model

## train and tune lda clasifier using caret
tune_params <- expand.grid(dimen = 0:10) # create tuning grid over available hyper parameters for NB model

bank_lda_tune <- train(x = trainingPredictors, y = trainingResponse, method = "lda2", trControl = train_control, tuneGrid = tune_params, metric = "Accuracy")
plot(bank_lda_tune)

bank_lda_tune$finalModel$tuneValue
bank_lda_finalHP <- bank_lda_tune$finalModel

rm(train_control, tune_params, bank_nb_tune, bank_lda_tune, train_index, trainingPredictors, trainingResponse)
## compare models over 10 differnt test/training data sets

train_control <- trainControl(method = "cv", number = 10)
tune_params_nb <- expand.grid(usekernel = bank_nb_finalHP$tuneValue[1,2], fL = bank_nb_finalHP$tuneValue[1,1], adjust = bank_nb_finalHP$tuneValue[1,3])
tune_params_lda <- expand.grid(dimen = bank_lda_finalHP$tuneValue[1,1])

all_nb <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors]
  trainingResponse <- modelData[train_index_10[ ,i],"Auth"]
  testData <- modelData[-train_index_10[,i], ]
  
  bank_mod4ass_nb <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params_nb)
  
  pred <- predict(bank_mod4ass_nb, newdata = testData)
  
  confusionMatrix(pred, testData$Auth)

})

all_lda <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors]
  trainingResponse <- modelData[train_index_10[ ,i],"Auth"]
  testData <- modelData[-train_index_10[,i], ]
 
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

performance<- data.frame(c(1:10), unlist(nb_acuracy), unlist(lda_acuracy), unlist(nb_Kappa), unlist(lda_Kappa), unlist(nb_Sens), unlist(lda_Sens), unlist(nb_Spec), unlist(lda_Spec))

names(performance) <- c("observation", "nb_acuracy","lda_acuracy", "nb_Kappa", "lda_Kappa", "nb_Sens", "lda_Sens", "nb_Spec", "lda_Spec")

rm(lda_acuracy , lda_Kappa, lda_Sens, lda_Spec, nb_acuracy, nb_Kappa, nb_Sens, nb_Spec)
