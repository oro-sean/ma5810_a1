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
plot_correlation(rawData[,1:4])

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index <- createDataPartition(rawData$Auth, p=0.8, list = FALSE) # returns numerical vector of the index of the observations to be included in the training set
predictors <- names(rawData[-5]) # return vector of column names, removing "Auth" as it is the response variable

testData <- rawData[-train_index,]

trainingPredictors <- rawData[train_index,predictors]
trainingResponse <- as.factor(rawData[train_index,"Auth"])


## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

## train and tune Naive Bayes classifier using caret

train_control <- trainControl(method = "cv", number = 10) # use 10 fold cross validation on the training set to asses model hyper parameters
tune_params <- expand.grid(usekernel = c(TRUE, FALSE), fL = 1:5, adjust = 1:5) # create tuning grid over available hyper parameters for NB model

bank_nb_mod1 <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params) # train model

plot(bank_nb_mod1) # plot accuracy across tuining grid

bank_nb_mod1$finalModel$tuneValue # return hyperparameters of "best" model

bank_nb_final <- bank_nb_mod1$finalModel # save the most acurate model


## train and tune lda clasifier using caret
train_control <- trainControl(method = "cv", number = 10)  # use 10 fold cross validation on the training set to asses model hyper parameters
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



## close cluster
stopCluster(cl)

