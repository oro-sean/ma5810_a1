
rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")
setwd("~/Documents/r_projects/a1")

library(naivebayes)
library(ggplot2)
library(caret, warn.conflicts = F, quietly = T)
library(dplyr)

set.seed(123)

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
rawData <- read.csv(file, header = FALSE)
names(rawData) <- c("variance", "skewness","kurtosis", "entropy", "auth")
rawData <- rawData %>% mutate(auth = ifelse(auth == 1, "True", "False"))
rawData$auth <- as.factor(rawData$auth)

test_index <- createDataPartition(rawData$auth, p=0.8, list = FALSE)
predictors <- names(rawData[-5])

testData <- rawData[test_index,]

trainingPredictors <- rawData[-test_index,predictors]
trainingResponse <- as.factor(rawData[-test_index,"auth"])

train_control <- trainControl(method = "cv", number = 10)
tune_params <- expand.grid(usekernel = c(TRUE, FALSE), fL = 0:5, adjust = 1:5)

bank_nb_mod1 <- train(x = trainingPredictors, y = trainingResponse, method = "nb", trControl = train_control, tuneGrid = tune_params)
confusionMatrix(bank_nb_mod1)
plot(bank_nb_mod1)
