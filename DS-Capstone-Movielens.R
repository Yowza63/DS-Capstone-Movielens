# Project MovieLens for HarvardX -  PH125.9x, Data Science: Capstone #
# This is a working file to flesh out code before incorporating in to the RMD file

# ************************************************************************************************
# Exploring the dataset - for the course quiz
# ************************************************************************************************

# load the libraries needed
library(tidyverse)

# read in the datasets produced from running MovieLens_Data_Wrangling.R
edx <- readRDS("movielens_training_data.rds")
validation <- readRDS("movielens_validation_data.rds")



# ************************************************************************************************
# Visualize the Data 
# ************************************************************************************************

# Make a smaller version of edx for experimentation
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.01, list = FALSE)
edx_small <- edx[test_index,]

# confirming there are no "NA" values in the data
for (i in 1:length(colnames(edx_small))){
  print(sum(is.na(edx_small[,1])))
}

# Examine structure of the data
str(edx_small)
sd(edx_small$rating)
summary(edx_small)

# Explore the data with a few plots
library(tidyverse)
library(ggplot2)
library(dplyr)

# 1. Number of ratings by year
tmp <- table(edx_small$year) # make a summary table of number of movies by year
x <- as.data.frame(tmp, optional = TRUE) %>% rename(Year = Var1, Number = Freq)
ggplot(x, aes(x = Year, y = Number)) + 
  geom_bar(stat = "identity", width = .8, fill = "steelblue") + 
  labs(title = "Number of Ratings", 
       x = NULL, 
       y = 'Number of Movie Ratings by Year') 

# 2. Movies released by year
# first confirm the number of unique movies
length(unique(edx_small$movieId))
# you can't create the data frame using this approach -- 
# tmp <- data.frame(movieId = unique(edx_small$movieId), title = unique(edx_small$movieId))
# while it will work, we can't reliably be sure the movieId and titles stay aligned

# create a temporary dataframe (tmp) of just movieId and title, remove duplicate rows, and then plot on a
# log scale as a dot plot
# The y scale has been transformed to a log 10 intentionally distorting the display of data so that we 
# can more clearly see the earlier values
# Interesting that the number of movies released per year caps out at around 441 in 2002 and tapers off to
# just 251 movies released in 2008
tmp <- edx %>% select(movieId, title) # create a table of just movieId and Title
tmp <- tmp[!duplicated(tmp$movieId),] # remove rows that are duplicate
tmp$release_year <- as.integer(str_sub(tmp$title, start = -5, end = -2)) # create the release_year column
tmp %>% group_by(release_year) %>% 
  summarize(Count = n()) %>% 
  ggplot(aes(x = release_year, y = Count)) + 
  scale_y_continuous(breaks=seq(0, 300, 50), trans = "log10") + 
  geom_point(alpha = .5, size = 2) + 
  labs(title = "Number of Movies Release by Year", 
       subtitle = "Shown on a logarithmic scale of 10", 
       x = NULL, 
       y = 'Number of Movies Released (Scale is Log 10)') 

a <- table(as.matrix(tmp$release_year, tmp$movieId))
max(a); min(a)

# 3.  avearge ratings per movie
# there does appear to be a slight positive correlation between the number of ratings and average
# this is borne out by the correleation of .22; more ratings imply a higher overall average rating
tmp <- edx %>% 
  group_by(movieId) %>% 
  summarize(n = n(), avg_rating = sum(rating)/n) %>%
  filter(n >= 100)

cor(tmp$avg_rating, tmp$n)

tmp %>% ggplot(aes(avg_rating, n)) + geom_point()

# 4. Table of individual genres
sapply(Genres, function(g) {
  sum(str_detect(edx$genres, g))
}) %>% 
  sort(decreasing = TRUE) %>%
  knitr::kable(., "simple", col.names = "Number of Ratings", format.args = list(big.mark = ","))

tmp <- edx %>% group_by(genres) %>% summarize(n = n())
tmp[order(-tmp$n),]

# 4. Are the ratings for the movies with the most ratings normally distributed? NOT FINISHED
tmp$movieId[which(tmp$n == max(tmp$n))]
a <- edx$rating[which(edx$movieId == 296)]

# 5. number of unique users per year
# the aggregate function, using n_distinct by year, is retuning the number of distinct items in 
# each year for every column. save as edx_distinct_by_year for use in later explorations
# Note:  I created a tiny sample of edx (.001% or 90 values) ran aggregate and then computed the
# expected values by pulling the data into excel using write.csv
edx_distinct_by_year <- aggregate(edx, by = list(edx$year), FUN = n_distinct)

edx_distinct_by_year %>% filter(Group.1 > 1995 & Group.1 < 2009) %>% 
  ggplot(aes(x = Group.1, y = userId)) + 
  geom_bar(stat = "identity", width = .8, fill = "steelblue") + 
  labs(title = "Number of Unique Users in Every Year", 
       x = NULL, 
       y = 'Number of Unique Users') 

# 6. average number of ratings per user - how active are users
length(edx$rating) / n_distinct(edx$userId) # the overall average

tmp <- edx %>% 
  group_by(userId) %>% 
  summarize(n = n(), ratings_per_user = sum(rating)/n) 

cor(tmp$ratings_per_user, tmp$n)

tmp %>% ggplot(aes(average_rating, n)) + geom_point()

length(edx$rating)/n_distinct(edx$userId) # what's the overall average

# Table or graph of genres, average rating, distinct users, number of ratings
tmp <- edx %>% group_by(genres) %>% summarize(n = n(), avg_rating = sum(rating)/n) %>% filter(n >= 10)
tmp[order(-tmp$avg_rating),]
tmp %>% ggplot(aes(n, avg_rating)) + geom_point()

# show the total rated for every user and plot the series
# NOT DONE!
library(dplyr)
tmp <- data.frame(id = edx_small$userId, rating = edx_small$rating) # dataset of just ratings and userId's
out <- aggregate(x = tmp, by = list(tmp$id), FUN = sum) %>% select(year = Group.1, id = id)
out2 <- aggregate(x = tmp, by = list(tmp$id), FUN = c('sum', 'length'))

# number of movies per genre (can we summarize the genres)?
# average ratings per genre
# distribution of ratings
# graph showing the normal curve and the distribution of ratings around that
# standard deviations and mean of average ratings by year
# standard deviations and mean of ratings by genre
# look for outliers in number of ratings per user and / or ratings per movie?

# goofing around

# Examine whether the data is normally distributed
# "The Q-Q plot, or quantile-quantile plot, is a graphical tool to help us assess if a set of data 
# plausibly came from some theoretical distribution such as a Normal or exponential.  For example, 
# if we run a statistical analysis that assumes our dependent variable is Normally distributed, we 
# can use a Normal Q-Q plot to check that assumption." 
# source:  https://data.library.virginia.edu/understanding-q-q-plots/

# A qq plot is a scatter plot of a theoretical normal distribution versus the actual distribution
# The quantiles or percentiles are the percent of data that falls below a specific percentage

# In case I want to user the validation data, which right now is really the training data (it's just big)
edx <- readRDS("movielens_training_data.rds")
validation <- readRDS("movielens_validation_data.rds")

# Histogram of the number of movies rated by each user
edx %>% group_by(userId) %>% summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(binwidth = 5, fill = "blue", col = "black") + 
  xlab("Number of Movies Rated") +
  ylab("Number of Users") + 
  labs(title = "Number of Movies Rated by Each User")

aa <- edx %>% group_by(userId) %>% summarize(n = n()) 
mean(aa$n)

a <- edx %>% group_by(userId) %>% summarize(n = n()) %>% filter(n <= 50)
mean(a$n)

sapply(Genres, function(g) {
  sum(str_detect(tmp$genres, g))
}) %>% 
  sort(decreasing = TRUE) %>%
  head(5) %>%
  knitr::kable(., "simple", col.names = "Number of Ratings", format.args = list(big.mark = ","))

dt <- mtcars[1:5, 1:6]
dt %>% 
  head(2) %>%
  kbl() %>%
  kable_styling()






# Save in case I want to add a column
# Create a year column to capture the year associated with the ratings (Becky added this code)
library(lubridate)
edx <- mutate(movielens, year = year(as_datetime(timestamp)))
validation <- mutate(movielens, year = year(as_datetime(timestamp)))


# Create a release_year column for the movie release year by extracting the information from the 
# title field (Becky added this code to that given by the course)
edx$release_year <- as.integer(str_sub(movielens$title, start = -5, end = -2))
validation$release_year <- as.integer(str_sub(movielens$title, start = -5, end = -2))

