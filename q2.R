## set up environment and import csv file

rm(list = ls()) # removes all variables
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console
setwd("~/Documents/GitHub/ma5810_a1") # set the working directory

# import required packages
library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(bnclassify)
library(DataExplorer)
library(doParallel)
library(reshape2)

set.seed(123) # sets seed for repeat ability of randomly generated numbers

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data' # store the path to the source data
rawData <- read.csv(file, header = FALSE, stringsAsFactors = TRUE) # import source data, in this case the data file has no headers and strings will be used a factors
names(rawData) <- c("Edible", "Cap-Shape", "Cap-Surface", "Cap-Colour", "Bruises", "Odour", "Gill-Attachment", "Gill-Spacing", "Gill-Size", "Gill-Colour", "Stalk-Shape", "Stalk-Root",  # name data.frame columns
                    "Stalk-Surface-Above-Ring", "Stalk-Surface-Below-Ring", "Stalk-Colour-Above-Ring", "Stalk-Colour-Below-Ring", "Veil-Type", "Veil-Colour", "Ring-Number", "Ring-Type", "Spore-Print-Colour", "Population", "Habitat")

## exploratory visualization of the data set
str(rawData) # returns the structure of the data.frame 
introduce(rawData) # returns some basic information about the data
plot_bar(rawData) # generate bar plots showing the count of each variable class

modelData <- within(rawData, rm("Veil-Type")) # drop "veil-Type" as it has only one class
plot_bar(modelData, by = "Edible") # create bar plots of all predictor variables showing the make up of the response variable in each class

# group minority classes and change variable type back to factor
modelData <- group_category(data = modelData, feature = "Cap-Colour", threshold = 0.2, update = TRUE) 
modelData <- group_category(data = modelData, feature = "Stalk-Colour-Below-Ring", threshold = 0.2, update = TRUE)
modelData <- group_category(data = modelData, feature = "Spore-Print-Colour", threshold = 0.2, update = TRUE)
modelData$`Cap-Colour` <- as.factor(modelData$`Cap-Colour`)
modelData$`Stalk-Colour-Below-Ring` <- as.factor(modelData$`Stalk-Colour-Below-Ring`)
modelData$`Spore-Print-Colour` <- as.factor(modelData$`Spore-Print-Colour`)

rm(rawData, file) # remove variables that will no longer be required

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index_10 <- createDataPartition(modelData$Edible, p=0.8, list = FALSE, times = 10) # returns numerical vector of the index of the observations to be included in the training set, repeat so we have 10 different test training sets for later
predictors <- names(modelData[-1]) # return vector of column names, removing "Auth" as it is the response variable

## set up training environment with the details of the validation method and tune grid
(modelLookup("nbDiscrete"))
train_control <- trainControl(method = "cv", number = 10) # instruct training to occur using k folds cross validation with 10 folds
tune_params_initial <- expand.grid(smooth = 1:10) # create a training grid

## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

start_time <- Sys.time() # record start time of model trainings

## tune model over all 10 test / training splits
all_nb_tune <- lapply(seq_len(ncol(train_index_10)), function(i){ # rather than use "for loop" lapply runs the defined function for all prescribed values of i
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors] # create data.frame of training predictors
  trainingResponse <- modelData[train_index_10[ ,i],"Edible"] # create training response vector
  
  mushroom_mod_tune <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_initial) # train model
  
  mushroom_mod_tune$finalModel$tuneValue[1,1] # return the hyperparameter for the best model found in the 10 fold cross validation

})

## get final hyper parameters
tune_params_final <- expand.grid(smooth = mean((unlist(all_nb_tune)))) # unlist and average hyperparameter from all 10 training sets

rm(all_nb_tune, tune_params_initial)

## train model across all 10 test / training splits and return accuracy of model each time
all_nb_accuracyTraining <- lapply(seq_len(ncol(train_index_10)), function(i){ # rather than use "for loop" lapply runs the defined function for all prescribed values of i
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors] # create data.frame of training predictors
  trainingResponse <- modelData[train_index_10[ ,i],"Edible"] # create training response vector
  mushroom_mod4ass <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_final) # train model
  
  mushroom_mod4ass$results["Accuracy"] # return the accuracy of the model on the training data
  
})

trainingAccuracy <- unlist(all_nb_accuracyTraining) # make training accuracy a simple vector showing the average model accuracy on the training data for each test/ train split 

## train model across all 10 test / training splits and return confusion matrix each time
all_nb_confusionMatrix <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors] # create data.frame of training predictors
  trainingResponse <- modelData[train_index_10[ ,i],"Edible"] # create training response vector
  testData <- modelData[-train_index_10[,i], ] # create test data set
  mushroom_mod4ass <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_final) # train model
  pred <- predict(mushroom_mod4ass, newdata = testData) # use model to make predictions using the predictors from the trainingset
  
  confusionMatrix(pred, testData$Edible) # return confusion matrix for predicted and actual classes
  
})

end_time <- Sys.time() # record time at end of analysis
runtime <- end_time - start_time # calculate run time


## build data.frame of the performance scores for each test/ training split

nb_acuracy <- lapply(seq_len(length(all_nb_confusionMatrix)), function(i){
  unlist(all_nb_confusionMatrix[[i]]$overall["Accuracy"])
})

nb_Kappa <- lapply(seq_len(length(all_nb_confusionMatrix)), function(i){
  unlist(all_nb_confusionMatrix[[i]]$overall["Kappa"])
})

nb_Sens <- lapply(seq_len(length(all_nb_confusionMatrix)), function(i){
  unlist(all_nb_confusionMatrix[[i]]$byClass["Sensitivity"])
})

nb_Spec <- lapply(seq_len(length(all_nb_confusionMatrix)), function(i){
  unlist(all_nb_confusionMatrix[[i]]$byClass["Specificity"])
})

performance <- data.frame(c(1:10), unlist(nb_acuracy), unlist(nb_Kappa), unlist(nb_Sens), unlist(nb_Spec), trainingAccuracy)
names(performance) <- c("Test/ Training Split #", "Accuracy","Kappa", "Sensitivity","Specitivity", "Accuracy on Training Data")

rm(nb_acuracy, nb_Kappa, nb_Sens, nb_Spec, all_nb_confusionMatrix)

## train model on single test training set to inspect the importance of individual variables and easily look into model results

trainingPredictors <- modelData[train_index_10[ ,1],predictors] # create data.frame of training predictors
trainingResponse <- modelData[train_index_10[ ,1],"Edible"] # create training response vector
testData <- modelData[-train_index_10[,1], ] # create test data set
mushroom_mod4ass <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_final) # train model
pred <- predict(mushroom_mod4ass, newdata = testData) # use model to make predictions using the predictors from the trainingset
confusionMatrix(pred, testData$Edible) # return confusion matrix for predicted and actual classes

## Plot model performance
performance_flat <- melt(performance, id.vars = 1) # flatten performance for easy use in ggplot
ggplot(data = performance_flat, aes(x =`Test/ Training Split #`, y = `value`, shape = `variable`, colour = `variable`)) + geom_point(size = 4) + geom_line() + theme(legend.position = "bottom") # plot each performance metric for each test / train split, with lines between points to easily identify trends

## plot plot variable importance
varimp <- varImp(mushroom_mod4ass)
plot(varimp)

## plot the 2 most and least important variable for discussion
spc <- ggplot(modelData) + aes(x = `Spore-Print-Colour`, fill = Edible) + geom_bar() + scale_fill_manual(values = c(e = "#23A904", p = "#E42511")) + theme_minimal()
gc <- ggplot(modelData) + aes(x = `Gill-Colour`, fill = Edible) + geom_bar() +scale_fill_manual(values = c(e = "#23A904", p = "#E42511")) + theme_minimal()
vc <- ggplot(modelData) + aes(x = `Veil-Colour`, fill = Edible) + geom_bar() + scale_fill_manual(values = c(e = "#23A904", p = "#E42511")) + theme_minimal()
ga <- ggplot(modelData) + aes(x = `Gill-Attachment`, fill = Edible) + geom_bar() + scale_fill_manual(values = c(e = "#23A904",  p = "#E42511")) + theme_minimal()
comparision <- ggarrange(spc, gc, vc, ga, labels = c("Spore-Print_colour","Gill-Colour", "Veil-Colour", "Gill-Attachment"), ncol = 2, nrow = 2) # group plots together, one on top of the other
annotate_figure(comparision, top = text_grob("Most and Least Important Variables", color = "blue", face = "bold", size = 16)) # give the plot frame a common tittle

## close cluster
stopCluster(cl)

