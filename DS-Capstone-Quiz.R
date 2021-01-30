# Project MovieLens for HarvardX -  PH125.9x, Data Science: Capstone #

# ****************************************************************
# Exploring the dataset - r code used to complete the course quiz
# ****************************************************************

# load the libraries needed
library(tidyverse)

# read in the datasets produced from running MovieLens_Data_Wrangling.R
edx <- readRDS("movielens_training_data.rds")
validation <- readRDS("movielens_validation_data.rds")

# Q1
dim(edx) # rows and columns, answer is 9000055 and 6, although my edx has 8 columns as I added 2

#Q2
sum(edx$rating == 0) # number of rows with a "0" rating
sum(edx$rating == 3) # number of rows with a "3" rating
edx %>% filter(rating == 0) %>% tally() # answer code, same answer but longer to run
edx %>% filter(rating == 3) %>% tally() # answer code, same answer but longer to run

# Q3
n_distinct(edx$movieId) # number of unique movies

# Q4
n_distinct(edx$userId) # number of unique users

# Q5 - number of ratings by major genre categories
edx %>% filter(str_detect(genres,"Drama")) %>% summarize(drama_num = n())
edx %>% filter(str_detect(genres,"Comedy")) %>% summarize(comedy_num = n())
edx %>% filter(str_detect(genres,"Thriller")) %>% summarize(thriller_num = n())
edx %>% filter(str_detect(genres,"Romance")) %>% summarize(romance_num = n())
# this produces the same results as the four lines above
genres = c("Action", "Drama", "Comedy", "Thriller", "Romance", "Documentary")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# finding the unique genres, not a question but helpful for later
a <- str_split(edx$genres, "\\|") %>% flatten_chr()
stri_unique(a)

# Q6 Find the movies with the most ratings
tmp <- edx %>% group_by(movieId, title) %>% summarize(num_ratings = n())
tmp[order(-tmp$num_ratings),]
# the answer code approach, avoids creating an object
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Q7 - What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>% arrange(desc(count))

# Q8 
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()