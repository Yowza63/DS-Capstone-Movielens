#####################################################################################################################
# Project MovieLens for HarvardX -  PH125.9x, Data Science: Capstone #
# Becky Johnson / Github: Yowza63
# R code to predict how users are likely to rate specific movies; from this the predictions for the highest rated
# movies can be recommended to a user as those they might enjoy
# Model is trained on the edx dataset and measured on the validation dataset using an RMSE loss calculation
# Full explanations and write-up are found in the file DS-Capstone-Movielens-Report.Rmd
#####################################################################################################################

# load the libraries needed
library(tidyverse)
library(caret)
library(stringi)
library(kableExtra)
library(ggplot2)
library(dplyr)
library(lubridate)
library(dlookr)
library(hrbrthemes)

# read in the datasets produced from running DS-Capstone-Movielens-Data.R
edx <- readRDS("movielens_training_data.rds")
validation <- readRDS("movielens_validation_data.rds")

###!!! Just for testing code !!!###
tmp <- edx[1:50000]

# ------------------------------------------------------------------------------------------------------------------
#  High level exploration of the data
# ------------------------------------------------------------------------------------------------------------------

# Confirm there are no "NA" values in the data using the apply function with the 2 argument for columns
apply(apply(edx, 2, is.na), 2, sum)

# Examine the structure of edx
str(edx)

# ------------------------------------------------------------------------------------------------------------------
#  User specific data exploration (userId)
# ------------------------------------------------------------------------------------------------------------------

# Do individual user's rate movies higher or lower, on average, than other user's?
# Do user's have different quantities of rated movies?
# Are there anomolies in the timing for how movies are rated?

# TODO: Consider adding columns for the average rating for each user and the total rated for each user

# How many unique users?
n_distinct(edx$userId)

# The average rating per user
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  summarize(avg_ratings_by_user = mean(avg_per_user)) %>%
  pull(avg_ratings_by_user)

# The overall average rating is .1 lower than the simple average of each user's average rating implying bias
mean(edx$rating)

# The user's average ratings appear normally distributed. There are clear user specific effects
edx %>% group_by(userId) %>% summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  ggplot(aes(avg_per_user)) + 
  geom_histogram(binwidth = .2, fill = "#69b3a2", col = "black") + 
  xlab("User Specific Average Ratings") +
  ylab("Number of Users") + 
  labs(title = "Average Ratings Across Users") + 
  theme_ipsum()

# Most users rate relatively few movies, but there is a long tail of users who rate thousands of films
edx %>% group_by(userId) %>% summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(binwidth = 25, fill = "#69b3a2", col = "black") + 
  xlab("Number of Movies Rated") +
  ylab("Number of Users") + 
  labs(title = "Volume of Movies Rated by User") + 
  theme_ipsum()

# Running the graph above but capping the population at those with 500 movies rated
edx %>% group_by(userId) %>% summarize(n = n()) %>% 
  filter(n <= 500) %>%
  ggplot(aes(n)) + 
  geom_histogram(binwidth = 25, fill = "#69b3a2", col = "black") + 
  xlab("Number of Movies Rated") +
  ylab("Number of Users") + 
  labs(title = "Volume of Movies Rated by User") + 
  theme_ipsum()

# What percent of users rate 25 or fewer movies?
edx %>% group_by(userId) %>% 
  summarize(n = n()) %>% 
  summarize(users_50_or_less = length(n[n <= 50])/length(n)) %>% 
  pull(users_50_or_less)

# What portion of ratings are associated with users rating 25 or fewer movies? Only 10%, so 43% of users
# represent just 10% of the total number of ratings
edx %>% group_by(userId) %>% 
  summarize(n = n()) %>% 
  summarize(ratings_users_50_or_less = sum(n[n <=50])/sum(n)) %>% 
  pull(ratings_users_50_or_less)

# Create a scatter plot of the data which shows outliers of users with large number of ratings and 
# lower overall averages
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating/n)) %>%
  ggplot(aes(avg_per_user, n)) + geom_point() + 
  xlab("Average Rating by User") + 
  ylab("Number of Movies Rated") + 
  labs(title = "Average Ratings and Volume Rated") + 
  theme_ipsum()

# Explore this further by looking at the averages for just those users with over 2000 ratings they clearly are 
# below the overall overage [TODO: LABEL THE VERTICAL LINE FOR THE OVERALL AVERAGE]
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating/n)) %>%
  filter(n >= 2000) %>%
  ggplot(aes(avg_per_user, n)) + geom_point() + 
  xlab("Average Rating by User") + 
  ylab("Number of Movies Rated") + 
  labs(title = "Average Ratings and Volume Rated", subtitle = "Users >= 2000 Movies Rated") + 
  geom_vline(xintercept = mean(edx$rating)) + 
  theme_ipsum()

# The averages for all users and those that rate 2000 or more movies are significantly different
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  filter(n >= 2000) %>%
  summarize(avg_ratings_by_user = mean(avg_per_user)) %>%
  pull(avg_ratings_by_user)

edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  summarize(avg_ratings_by_user = mean(avg_per_user)) %>%
  pull(avg_ratings_by_user)

# To visualize the differences between users with fewer ratings and those who've rated thousands we'll create bins
# of the data by number of ratings and create a boxplot for each bin. The plot shows that the largest population
# of users are those who only rate a few movies and these users tend to have higher ratings.
# TODO:  Change the bin names to something more meaningful, add a label for the number of users in each bin
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>%
  mutate(bin = binning(n, nbins = 6, type = "equal")) %>% 
  ggplot(aes(x=bin, y=avg_per_user) ) +
  geom_boxplot(fill="#69b3a2") + 
  theme_ipsum() +
  xlab("Bin") + 
  theme(axis.text.x = element_text(face="plain", color="#69b3a2", size=8, angle=45))

# TODO: compute the average rating for each user and total number of movies rated, sort lowest to 
# highest by number rated, split into 10 buckets, compute the average ratings, % of total ratings, and
# total number of ratings for each bucket
# Can the aggregate function help?

# ------------------------------------------------------------------------------------------------------------------
#  Explore the timing of when user's rated movies (timestamp)
# ------------------------------------------------------------------------------------------------------------------
# What is the average length of time a user has been rating movies?
# What is the distribution of time rating movies?
# Are they anomolies in how many movies are rated in a given day or week?
# Do users who have been rating movies longer tend to have lower ratings?

a <- tmp %>% group_by(userId) %>% summarize(n=n(), userId=userId, timestamp = timestamp)
a <- a %>% group_by(userId) %>% 
  mutate(last_rating_date = max(timestamp), first_rating_date = min(timestamp))
a <- a %>% select(-timestamp)
index <- which(duplicated(a))
a <- a[-index,]

year(as_datetime(a$last_rating_date)) - year(as_datetime(a$first_rating_date))

b <- as_datetime(tmp$timestamp[1000])
c <- as_datetime(tmp$timestamp[5000])

a <- tmp %>% mutate(date = as_datetime(timestamp)) %>%
  group_by(userId) %>%
  mutate(last_rating_date = max(date), first_rating_date = min(date)) %>%
  summarize(n = n())



year(as_datetime(tmp$timestamp))

as.data.frame(strsplit(as_datetime(tmp$timestamp), " "))

tmp <- mutate(tmp, year = year(as_datetime(timestamp)))


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



# Movies by year
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

# Table or graph of genres, average rating, distinct users, number of ratings
tmp <- edx %>% group_by(genres) %>% summarize(n = n(), avg_rating = sum(rating)/n) %>% filter(n >= 10)
tmp[order(-tmp$avg_rating),]
tmp %>% ggplot(aes(n, avg_rating)) + geom_point()

# show the total rated for every user and plot the series
# NOT DONE!
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

edx <- mutate(movielens, year = year(as_datetime(timestamp)))
validation <- mutate(movielens, year = year(as_datetime(timestamp)))


# Create a release_year column for the movie release year by extracting the information from the 
# title field (Becky added this code to that given by the course)
edx$release_year <- as.integer(str_sub(movielens$title, start = -5, end = -2))
validation$release_year <- as.integer(str_sub(movielens$title, start = -5, end = -2))

