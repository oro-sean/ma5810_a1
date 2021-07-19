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

file <- 'HBblood.csv'
rawData <- read.csv(file, header = TRUE, stringsAsFactors = TRUE)