## set up environment and import csv file

rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")
setwd("~/Documents/GitHub/ma5810_a1")

library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(bnclassify)
library(DataExplorer)
library(doParallel)
library(esquisse)

set.seed(123)

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
rawData <- read.csv(file, header = FALSE, stringsAsFactors = TRUE)
names(rawData) <- c("Edible", "Cap-Shape", "Cap-Surface", "Cap-Colour", "Bruises", "Odour", "Gill-Attachment", "Gill-Spacing", "Gill-Size", "Gill-Colour", "Stalk-Shape", "Stalk-Root", 
                    "Stalk-Surface-Above-Ring", "Stalk-Surface-Below-Ring", "Stalk-Colour-Above-Ring", "Stalk-Colour-Below-Ring", "Veil-Type", "Veil-Colour", "Ring-Number", "Ring-Type", "Spore-Print-Colour", "Population", "Habitat")


## exploratory visualization of the dataset
str(rawData)
introduce(rawData)
head(rawData)
summary(rawData)
plot_bar(rawData)

modelData <- rawData

## create test training split, create test data frame and create training predictors data frame and training response vector
train_index_10 <- createDataPartition(modelData$Edible, p=0.8, list = FALSE, times = 10) # returns numerical vector of the index of the observations to be included in the training set, repeat so we have 10 different test training sets for later
predictors <- names(modelData[-1]) # return vector of column names, removing "Auth" as it is the response variable

rm(rawData, file)

train_control <- trainControl(method = "cv", number = 10)
tune_params_initial <- expand.grid(smooth = 1:10)

## register doParallel
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

start_time <- Sys.time()

all_nb_tune <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors]
  trainingResponse <- modelData[train_index_10[ ,i],"Edible"]
  testData <- modelData[-train_index_10[,i], ]
  
  mushroom_mod_tune <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_initial)
  
  mushroom_mod_tune$finalModel$tuneValue[1,1]

})

tune_params_final <- expand.grid(smooth = mean((unlist(all_nb_tune))))

all_nb <- lapply(seq_len(ncol(train_index_10)), function(i){
  
  trainingPredictors <- modelData[train_index_10[ ,i],predictors]
  trainingResponse <- modelData[train_index_10[ ,i],"Edible"]
  testData <- modelData[-train_index_10[,i], ]
  
  mushroom_mod4ass <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_final)

  pred <- predict(mushroom_mod4ass, newdata = testData)
  
  confusionMatrix(pred, testData$Edible)
  
})

end_time <- Sys.time()
runtime <- end_time - start_time
## close cluster
stopCluster(cl)


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

performance<- data.frame(c(1:10), unlist(nb_acuracy), unlist(nb_Kappa), unlist(nb_Sens), unlist(nb_Spec))

names(performance) <- c("observation", "nb_acuracy","nb_Kappa", "nb_Sens","nb_Spec")

rm(nb_acuracy, nb_Kappa, nb_Sens, nb_Spec)

trainingPredictors <- modelData[train_index_10[ ,1],predictors]
trainingResponse <- modelData[train_index_10[ ,1],"Edible"]
testData <- modelData[-train_index_10[,1], ]

mushroom_mod <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control, tuneGrid = tune_params_final)

