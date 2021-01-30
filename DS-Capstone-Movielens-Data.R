# Project MovieLens for HarvardX -  PH125.9x, Data Science: Capstone #
# Author: Becky Johnson
# Github User: Yowza63
# This file pulls the data from the grouplens.org site, creates a training set (edx) and a testing
# set (validation). These are stored in movielens_training_data.rds and movielens_validation_data.rds for later

# ************************************************************************************
# Create edx set, validation set (final hold-out test set)
# Note: this process could take a couple of minutes
# ************************************************************************************

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

## Create the edx set (to work with and create our models) and the validation set
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set; this leaves a set, validation, that is 
# 8 rows fewer than temp. We're trying to predict how user's will rate movies so we want to build a model with
# data for specific users and then try that out on movies that user has rated in the validation set. So the 
# users in the validation set need to all be in the test set.
# The semi_join function keeps the rows in validation that have movieId and userId's that match items in edx
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set (this is just 8 rows I think)
# returns rows from temp where there are not matching values in validation, keeping just columns in temp
removed <- anti_join(temp, validation) 
edx <- rbind(edx, removed)

# save the edx and validation objects to a file to reference from the main RMD report
saveRDS(edx, 'movielens_training_data.rds')
saveRDS(validation, 'movielens_validation_data.rds')

# removes elements used to create needed data
rm(dl, ratings, movies, test_index, temp, movielens, removed, edx, validation) 



