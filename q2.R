## set up environment and import csv file

rm(list = ls())
if(!is.null(dev.list())) dev.off()
cat("\014")
setwd("~/Documents/GitHub/ma5810_a1")

library(naivebayes)
library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(bnclassify)

set.seed(123)

file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
rawData <- read.csv(file, header = FALSE, stringsAsFactors = TRUE)
names(rawData) <- c("Edible", "Cap-Shape", "Cap-Surface", "Cap-Colour", "Bruises", "Odour", "Gill-Attachment", "Gill-Spacing", "Gill-Size", "Gill-Colour", "Stalk-Shape", "Stalk-Root", 
                    "Stalk-Surface-Above-Ring", "Stalk-Surface-Below-Ring", "Stalk-Colour-Above-Ring", "Stalk-Colour-Below-Ring", "Veil-Type", "Veil-Colour", "Ring-Number", "Ring-Type", "Spore-Print-Colour", "Population", "Habitat")

head(rawData)
str(rawData)

## training test split

train_index <- createDataPartition(rawData$Edible, p=0.8, list = FALSE)
predictors <- names(rawData[-1])

testData <- rawData[-train_index,]

trainingPredictors <- rawData[train_index,predictors]
trainingResponse <- as.factor(rawData[train_index,"Edible"])

NB_Mushrooms <- naive_bayes(x = trainingPredictors, y = trainingResponse)
Pred_class <- predict(NB_Mushrooms, newdata = testData[,2:21], type = "class")
cont_tab <- table(Pred_class, testData$Edible)
sum(diag(cont_tab))/sum(cont_tab)
plot(NB_Mushrooms)

## Train NB Classier

train_control <- trainControl(method = "cv", number = 10)


edible_nb_mod1 <- train(x = trainingPredictors, y = trainingResponse, method = "nbDiscrete", trControl = train_control)
confusionMatrix(edible_nb_mod1$finalModel)
