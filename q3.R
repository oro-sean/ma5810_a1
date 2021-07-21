## set up environment and import data file

rm(list = ls()) # removes all variables
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console
setwd("~/Documents/GitHub/ma5810_a1") # set the working directory

# import required packages 
library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(DataExplorer)
library(doParallel)
library(esquisse)
library(reshape2)

set.seed(123) # sets seed for repeat ability of randomly generated numbers

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt' # store the path to the source data
rawData <- read.csv(file, header = FALSE) # import source data, in this case the data file has no headers and strings will be used a factors
names(rawData) <- c("Variance", "Skewness","Kurtosis", "Entropy", "Auth") # set column names in data.frame
rawData <- rawData %>% mutate(Auth = ifelse(Auth == 1, "Authentic", "Fraudulent")) # change the values Auth with an observation of 1 to "True", otherwise "False"
rawData$Auth <- as.factor(rawData$Auth) # make the "Auth" column a factor

## exploratory visualization of the dataset and summation of data set
str(rawData)
introduce(rawData)
head(rawData)
summary(rawData)
plot_histogram(rawData)
plot_bar(rawData)
plot_qq(rawData)
plot_correlation(rawData[ ,1:4])

modelData <- rawData # the raw data set looks suitable to model with

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index_10 <- createDataPartition(modelData$Auth, p=0.8, list = FALSE, times = 10) # returns numerical vector of the index of the observations to be included in the training set, repeat so we have 10 different test training sets for later
train_index <- train_index_10[ ,1] # tuining hyperparaemters over a single test set is suitable so split out first index
predictors <- names(modelData[-5]) # return vector of column names, removing "Auth" as it is the response variable

testData <- modelData[-train_index, ] # create data.frame of test data
trainingPredictors <- modelData[train_index,predictors] # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index, "Auth"]) # create vector of training responses

rm(rawData, file) # remove unused variables

## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

## train and tune Naive Bayes classifier using caret. We will first tune both models using a single test and training data set to obtain the best hyperparameters for each model
start_time <- Sys.time() # record start time

train_control <- trainControl(method = "cv", number = 10) # use 10 fold cross validation on the training set to asses model hyper parameters
tune_params <- expand.grid(usekernel = c(TRUE, FALSE), fL = 1:5, adjust = 1:5) # create tuning grid over available hyper parameters for NB model

bank_nb<- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params, metric = "Accuracy") # train model
bank_nb_final <- bank_nb$finalModel # save the most acurate model

tune_params <- expand.grid(dimen = 0:10) # create tuning grid over available hyper parameters for LDA model

bank_lda <- train(x = trainingPredictors, y = trainingResponse, method = "lda2", trControl = train_control, tuneGrid = tune_params, metric = "Accuracy") # train model
bank_lda_final <- bank_lda$finalModel # save most accurate model

rm(tune_params, train_index, trainingPredictors, trainingResponse)

## create confusion matrix to summarize each model
pred <- predict(bank_nb, newdata = testData) # use model to make predictions using the predictors from the trainingset
confusionMatrix(pred, testData$Auth) # return confusion matrix for predicted and actual classes
pred <- predict(bank_lda, newdata = testData) # use model to make predictions using the predictors from the trainingset
confusionMatrix(pred, testData$Auth) # return confusion matrix for predicted and actual classes

rm(pred) # remove unused variables

## compare models over 10 different test/training data sets
tune_params_nb <- expand.grid(usekernel = bank_nb_final$tuneValue[1,2], fL = bank_nb_final$tuneValue[1,1], adjust = bank_nb_final$tuneValue[1,3]) # set the model hyper parameters to the values that were best from the above step
tune_params_lda <- expand.grid(dimen = bank_lda_final$tuneValue[1,1]) # set the model hyper parameters to the values that were best from the above step

all_nb <- lapply(seq_len(ncol(train_index_10)), function(i){ # rather than use "for loop" lapply runs the defined function for all prescribed values of i
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors] # create data.frame of training predictors
  trainingResponse <- modelData[train_index_10[ ,i],"Auth"] # create training response vector
  testData <- modelData[-train_index_10[,i], ] # create data.frame of tests data
  bank_mod4ass_nb <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params_nb) # train model
  pred <- predict(bank_mod4ass_nb, newdata = testData) # use model to create predictions
  
  confusionMatrix(pred, testData$Auth) # return confusion matrix

})

all_lda <- lapply(seq_len(ncol(train_index_10)), function(i){ # rather than use "for loop" lapply runs the defined function for all prescribed values of i
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors] # create data.frame of training predictors
  trainingResponse <- modelData[train_index_10[ ,i],"Auth"] # create training response vector
  testData <- modelData[-train_index_10[,i], ] # create data.frame of tests data
  bank_mod4ass_lda <- train(x = trainingPredictors, y = trainingResponse, method = "lda2", trControl = train_control, tuneGrid = tune_params_lda) # train model
  pred <- predict(bank_mod4ass_lda, newdata = testData) # use model to create predictions
  
  confusionMatrix(pred, testData$Auth) # return confusion matrix
  
})

end_time <- Sys.time() # record end tiem of models
(runtime <- end_time - start_time) # calculate run time


## close cluster
stopCluster(cl)

## unpacked the performance metrics from the LDA and NB confusion matrices

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

performance<- data.frame(c(1:10), unlist(nb_acuracy), unlist(lda_acuracy), unlist(nb_Kappa), unlist(lda_Kappa), unlist(nb_Sens), unlist(lda_Sens), unlist(nb_Spec), unlist(lda_Spec)) # create data frame of performance metrics

names(performance) <- c("Test/ Training Split #", "nb_acuracy","lda_acuracy", "nb_Kappa", "lda_Kappa", "nb_Sens", "lda_Sens", "nb_Spec", "lda_Spec") # name columns of data frame

rm(lda_acuracy , lda_Kappa, lda_Sens, lda_Spec, nb_acuracy, nb_Kappa, nb_Sens, nb_Spec) # remove unused variables

## Plot comparison of model performance
performance_reduced <-performance[ , c(1,2,3,6,7)] 
performance_flat <- melt(performance_reduced, id.vars = 1) # flatten performance for easy use in ggplot
ggplot(data = performance_flat, aes(x =`Test/ Training Split #`, y = `value`, shape = `variable`, colour = `variable`)) + geom_point(size = 4) + geom_line() + theme(legend.position = "bottom") # plot each performance metric for each test / train split, with lines between points to easily identify trends

## Plot correlation for discusions regarding assumptions
plot_correlation(modelData[ ,1:4]) # create correlation heat map
