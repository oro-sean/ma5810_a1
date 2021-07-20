## set up environment and import data file

rm(list = ls()) # removes all variables
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console
setwd("~/Documents/GitHub/ma5810_a1") # set the working directory

# import required packages 
library(DataExplorer)
library(ggplot2)
library(ggpubr)
library(esquisse)

set.seed(123) # sets seed for repeat ability of randomly generated numbers

file <- 'HBblood.csv' # in this case the data is stored as a ".csv" file in the woking directory
rawData <- read.csv(file, header = TRUE, stringsAsFactors = TRUE) # the data is imported, the .csv file has headers which will be used as the column names of the data frame, any stirngs in the data set will be treated as factors

## data exploration and visualization
str(rawData) # returns the structure of the data.frame 
introduce(rawData) # returns some basic information about the data
head(rawData) # shows the first 6 lines of data
summary(rawData) # returns a basic statistical summary of the data

ggplot(rawData) + aes(x = SBP, y = HbA1c, colour = Ethno) + geom_point(shape = "circle", size = 1.5) + scale_color_hue(direction = 1) + theme_minimal() # create scatter plot with predictor variables on x and y, colour points by Ethno

hb <- ggplot(rawData) + aes(x = HbA1c) + geom_density(adjust = 1L, fill = "#FF8C00") + theme_minimal() # create distribution plot of HbA1c
sb <- ggplot(rawData) + aes(x = SBP) + geom_density(adjust = 1L, fill = "#EF562D") + theme_minimal() # create distribution plot of SBP
densityPlot <- ggarrange(hb,sb, labels = c("HbA1c", "SBP"), ncol = 1, nrow = 2) # group plots together, one on top of the other
annotate_figure(densityPlot, top = text_grob("Density Distributions of Predictor variables HbA1c and SBP", color = "blue", face = "bold", size = 16)) # give the plot frame a common tittle

plot_correlation(rawData[ ,2:3], title = "Correlation Between Predictor Variables") # create a correlation plot to asses correlation and covariance between predictors

hbByGroup <- ggplot(rawData) + aes(x = HbA1c, fill = Ethno) + geom_density(adjust = 1L) + scale_fill_hue(direction = 1) + theme_minimal() # create density plot by groups for HbA1c
sbpByGroup <- ggplot(rawData) + aes(x = SBP, fill = Ethno) + geom_density(adjust = 1L) + scale_fill_hue(direction = 1) + theme_minimal() # create density plot by groups for SBP
groupeddensityPlot <- ggarrange(hbByGroup,sbpByGroup, labels = c("HbA1c", "SBP"), ncol = 1, nrow = 2) # group plots together, one on top of the other
annotate_figure(groupeddensityPlot, top = text_grob("Density Distributions of Predictor variables Grouped by Ethno", color = "blue", face = "bold", size = 16)) # give the plot frame a common tittle



