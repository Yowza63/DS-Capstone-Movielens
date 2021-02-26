##################################################################################################################
# Project MovieLens for HarvardX -  PH125.9x, Data Science: Capstone #
# Becky Johnson / Github: Yowza63
# R code to predict how users will rate specific movies using the MovieLens 10M dataset from grouplens.org
# Model is trained on the edx dataset and measured on the validation dataset using an RMSE loss calculation
# Full explanations and write-up are found in the file DS-Capstone-Movielens-Report.Rmd
##################################################################################################################

# ---------------------------------------------------------------------------------------------------------------
#  STEP 1: Create the Datasets
#  Download files from the grouplens.org site, create a training set (edx) and a testing set (validation). 
#  These are stored in movielens_training_data.rds and movielens_validation_data.rds to avoid re-running these
#  step during code development
# ---------------------------------------------------------------------------------------------------------------

# Create edx set, validation set (final hold-out test set)
# Note: this process could take several minutes

# Load the needed libraries for this step
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# For reference the urls for the MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Process the data
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

# Use semi_join() to remove users from temp (soon to be the validation set) that are not in edx. This results
# in removing 8 rows. The rationale is that we are trying to predict how user's will rate movies so we want to 
# build a model with data for specific users and then try that out on movies that user has rated in the 
# validation set. So the users in the validation set need to all be in the test set.
# The semi_join function keeps the rows x (temp) that have movieId and userId's that match items in y (edx)
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

anti_join(temp, validation, by = "movieId") # shows the rows that have been removed

# Add rows removed from validation set back into edx set
# returns rows from temp where there are not matching values in validation, keeping just columns in temp
removed <- anti_join(temp, validation) 
edx <- rbind(edx, removed)

# save the edx and validation objects to a file to reference if working on the project over time
saveRDS(edx, 'movielens_training_data.rds')
saveRDS(validation, 'movielens_validation_data.rds')

# removes elements used to create needed data
rm(dl, ratings, movies, test_index, temp, movielens, removed, edx, validation) 

# ---------------------------------------------------------------------------------------------------------------
#  STEP 2:  Read in the previously processed and stored datasets
#  Note: Start here if the first step has previously been run
# ---------------------------------------------------------------------------------------------------------------
edx <- readRDS("movielens_training_data.rds")
validation <- readRDS("movielens_validation_data.rds")

# ---------------------------------------------------------------------------------------------------------------
#  STEP 3: Load libraries used in the analysis
# ---------------------------------------------------------------------------------------------------------------

# Load the libraries needed for the data exploration and model fitting

# Data science tools
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# Streamlines model training process for complex regression and classification problems
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Character string processing
if(!require(stringi)) install.packages("stringi", repos = "http://cran.us.r-project.org")
# Build html friendly tables
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
# Graphing tools
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
# Data manipulation tools
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
# Working with dates
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
# Data exploration
if(!require(dlookr)) install.packages("dlookr", repos = "http://cran.us.r-project.org")
# Formatting for graphs
if(!require(hrbrthemes)) install.packages("hrbrthemes", repos = "http://cran.us.r-project.org")
# Includes the helpful percent.table function
if(!require(sur)) install.packages("sur", repos = "http://cran.us.r-project.org")
# Matrix factoization tool
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# ---------------------------------------------------------------------------------------------------------------
#  STEP 4: Exploration of movielens data captured in edx and validation
# ---------------------------------------------------------------------------------------------------------------

# Confirm there are no "NA" values in the data using the apply function with the 2 argument for columns
apply(apply(edx, 2, is.na), 2, sum)

# Examine the structure of edx
str(edx)
str(validation)

# Number of rows and columns
dim(edx)
dim(validation)

# Columns, shows there are 6 columns in the dataset
colnames(edx)
summary(edx)

# Range of ratings, shows a tendency towards higher ratings
ratings_range <- percent.table(edx$rating)

# ---------------------------------------------------------------------------------------------------------------
#  STEP 4a:  User specific data exploration (userId)
# ---------------------------------------------------------------------------------------------------------------

# How many unique users?
n_distinct(edx$userId)

# The average rating per user
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  summarize(avg_ratings_by_user = mean(avg_per_user)) %>%
  pull(avg_ratings_by_user)

# The overall average rating is .1 lower than the simple average of each user's average rating implying bias
mean(edx$rating)

# The user's average ratings appear normally distributed
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

# mannual method of finding the bins to create the labels
c <- edx %>% group_by(userId) %>%
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% 
  filter(n <= 1000) %>%
  mutate(bin = binning(n, nbins = 10, type = "equal"))

# show the bins and the number of user's in each bin
knitr::kable(table(c$bin), format="markdown", digits = 3)
knitr::kable(percent.table(c$bin), format="markdown", digits = 1)

# create a chart showing a boxplot for every bin - shows the tendency toward lower ratings as user's rate more movies
edx %>% group_by(userId) %>% 
  summarize(n = n(), avg_per_user = sum(rating)/n) %>% # create the metrics for each user
  filter(n <= 1000) %>% # remove the outliers of user's with thousands of ratings
  mutate(bin = binning(n, nbins = 10, type = "equal")) %>% # create 10 bins for 1-1000 ratings
  ggplot(aes(x=bin, y=avg_per_user) ) + # plot the bins
  geom_boxplot(fill="#69b3a2") + # add green to the boxplots
  theme_ipsum() + # format for the graph
  ylab("Average Rating Per User") + 
  labs(title = "Number of Movies Rated", subtitle = "Excludes Users with > 1000 Ratings") + 
  # format the fonts and size of the labels
  theme(axis.text.x = element_text(face="plain", color="black", size=8, angle=90), 
        plot.title = element_text(size = 14), plot.subtitle = element_text(size = 10)) + 
  # add nicer labels for the x tickmarks
  scale_x_discrete("Number of Ratings", labels = c("1-109", "109-208", "208-307", "307-406", "406-505", 
                                                   "505-604", "604-703", "703-802", "802-901", "901-1000"))

# ---------------------------------------------------------------------------------------------------------------
#  STEP 4b: Explore the timing of when user's rated movies (timestamp)
# ---------------------------------------------------------------------------------------------------------------

# Use timestamp field to create a date of first rating and date of last rating for each distinct user and 
# save in a new dataframe, edx_time. NOTE: This takes a minute or so.
edx_time <- edx %>% 
  group_by(userId) %>% 
  summarize(n=n(), userId=userId, timestamp = as_datetime(timestamp), 
       avg_per_user = sum(rating)/n) %>% 
  mutate(last_rating_date = max(timestamp), first_rating_date = min(timestamp)) %>% # add first and last date 
  select(-timestamp) %>% # drop the timestamp column as it's no longer needed
  distinct() %>% # remove duplidate rows
  mutate(time_rating = difftime(last_rating_date, first_rating_date, units=c("secs"))) # length of time rating

# confirm the number of rows in edx_time is equal to the distinct number of users
n_distinct(edx$userId) == n_distinct(edx$userId)

# Show the time versus cumulative distribution for the length of time user's have been rating movies
edx_time %>% 
  ggplot(aes(x=as.numeric(time_rating/(60*24*60)))) + 
  stat_ecdf() +
  labs(title = "Cumulative Distribution of Time Rating", subtitle = "Shown in Days") + 
  ylab("Percent of Users") + 
  xlab("Duration of Time Ratings Were Logged") + 
  theme(plot.title = element_text(size = 12), 
        axis.text.x = element_text(face="plain", color="black", size=8, angle=90)) + 
  theme_ipsum()

# Compute the portion of users who've rated movies less than 1 week, less than 1 day, and less than 180 days
sum(edx_time$time_rating < (24*60*60)) / length(edx_time$time_rating) # less than 1 day
sum(edx_time$time_rating < (24*60*60*7)) / length(edx_time$time_rating) # less than 1 week
sum(edx_time$time_rating < (24*60*60*180)) / length(edx_time$time_rating) # less than 180 days

# Create bins or groups for the length of time users have been rating movies
tmp <- cut(as.numeric(edx_time$time_rating), 
           breaks=c(0, 24*60*60, 7*24*60*60, 30*24*60*60, 365*24*60*60, 2 * 365*24*60*60, Inf), 
           labels = c("Within 1 Day", ">1 Day to 1 Week", ">1 Week to 1 Month", 
                      ">1 Month to 1 Year", "> 1 Year to < 2 Years", ">2 Years"), 
           include.lowest = TRUE)
newtmp <- data.frame(time_rating = edx_time$time_rating, tmp = tmp, 
                     avg_per_user = edx_time$avg_per_user)

# Plot the bins which shows the average isn't really different
newtmp %>% ggplot(aes(x=tmp, y=avg_per_user) ) + # plot the buckets
  geom_boxplot(fill="#69b3a2") + # add green to the boxplots
  theme_ipsum() + # format for the graph
  ylab("Average Rating Per User") + 
  labs(title = "Span of Time User's Rated Movies") + 
  # format the fonts and size of the labels
  theme(axis.text.x = element_text(face="plain", color="black", size=8, angle=45), 
        plot.title = element_text(size = 14), plot.subtitle = element_text(size = 10)) + 
  # add nicer labels for the x tickmarks
  scale_x_discrete("Range of Time", labels = c(list(levels(tmp))))

# Looking at the averages confirms the consistency across the different groups. 
newtmp %>% group_by(tmp) %>% summarize(avg = mean(avg_per_user))

# ---------------------------------------------------------------------------------------------------------------
#  STEP 4c:  Explore movie specific effects (movieId, title)
# ---------------------------------------------------------------------------------------------------------------

# The number of distinct movies
n_distinct(edx$movieId)

# The scatterplot of average ratings and volumne of ratings (n) shows a positive relationship between 
# number of ratings and average ratings. It also shows a large number of movies with very few ratings 
edx_movies %>% 
  ggplot(aes(x = n, y = avg_rating)) + 
  geom_point() + 
  theme_ipsum() + 
  ylab("Average Rating") + 
  xlab("Number of Times Rated") + 
  labs(title = "Num. of Ratings per Movie vs Average Rating") + 
  theme(axis.text.x = element_text(face="plain", color="black", size=8, angle=45), 
        plot.title = element_text(size = 14), plot.subtitle = element_text(size = 10))

# We can confirm there is a positive correlation between the number of times a movie is rated (n) and 
# the average rating
cor(edx_movies$avg_rating, edx_movies$n)

# ---------------------------------------------------------------------------------------------------------------
#  STEP 4d:  Explore genre data
# ---------------------------------------------------------------------------------------------------------------

# The number of distinct genres
n_distinct(edx$genres)

# Find the most popular genres
edx %>% group_by(genres) %>% summarize(n = n()) %>% arrange(desc(n)) %>% head(10)

# Find the distinct genres
genres <- str_split(edx$genres, "\\|") %>% flatten_chr() %>% stri_unique()

# Show the distinct genres with frequency
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
}) %>% 
  sort(decreasing = TRUE) %>%
  knitr::kable(., "simple", col.names = "Number of Ratings", format.args = list(big.mark = ","))

# Explore the average rating per genre. Create a histogram of genres by average rating or boxplots?
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating)) %>%
  filter(n >= 1000) %>% # remove outliers of infrequently rated movies
  ggplot(aes(x = avg)) + 
  geom_histogram(binwidth = .25, fill = "#69b3a2", col = "black") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  theme_ipsum() + 
  labs(title = "Average Rating by Genres") + 
  xlab("Average Rating") + 
  ylab("Number of Genres")

# ---------------------------------------------------------------------------------------------------------------
#  STEP 5: Fitting the model
# ---------------------------------------------------------------------------------------------------------------

# Create a test and training set from edx
set.seed(755)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Remove these entries that appear in both, using the `semi_join` function:
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Define the loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# ---------------------------------------------------------------------------------------------------------------
#  STEP 5a: A simple approach
# ---------------------------------------------------------------------------------------------------------------

# Define a simple model just using the overall average rating and name it "Just the average". In this
# model every movie is rated at the overall average by every user
mu <- mean(train_set$rating)
mu

naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

# Create a table to store the results
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# Add the movie effects recognizing that some movies are rated higher on average than others
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# The histogram shows that the estimates have a wide range of variability
movie_avgs %>% ggplot(aes(b_i)) + 
  geom_histogram(binwidth = .5, fill = "#69b3a2", col = "black") +
  theme_ipsum() + 
  labs(title = "Movie Effects", subtitle = "Average Movie Rating vs Overall Average Rating") + 
  xlab("Movie Avg - Overall Avg") + 
  ylab("Number of Movies")

# Compute the RMSE for adding the effect of the average rating for movieId
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE(predicted_ratings, test_set$rating)

# Add these results
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie Effect Model", RMSE = model_1_rmse))

# Compute the average rating for user userId for those that have rated over 100 movies
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(binwidth = .25, fill = "#69b3a2", col = "black") + 
  theme_ipsum()
# TODO: Add titles for this graph

# A model to account for individual user bias
# First create a data set of the user specific effects defined as the overall avg rating for that 
# user minus the overall average rating across all users minus the variability associated with a 
# particular movie
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>% # adds b_i to the train_set
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i)) 

# Create predicted ratings using the user effect, movie effect and overall average
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
# Reset ratings which are >5 to 5 as that is the highest possible rating
predicted_ratings <- ifelse(predicted_ratings > 5, 5, predicted_ratings)

# Compute the RMSE of this new model with both user and movie effects and save in an object
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
# Add the results to our table
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User Effects Model", 
      RMSE = model_2_rmse))

# ---------------------------------------------------------------------------------------------------------------
#  STEP 5b: Using regularization
# ---------------------------------------------------------------------------------------------------------------
# Look at largest errors of our model, so-far - evidence of peculiarities in the data related to 
# movies with very few ratings
test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  pull(title)

# best and worst movies
# a table of just movie titles
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# best movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10)  %>% 
  pull(title)

# worst
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10)  %>% 
  pull(title)

# look at how frequently the best and worst are rated
train_set %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)

# Regularization permits us to penalize estimates that are formed using small sample sizes. It 
# has commonalities with the Bayesian approach 
# Penalized least squares. Compute these regularized estimates of $b_i$ using lamda = 2
lambda <- 2
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Relook at the top 10 and bottom 10 which now make more sense!
train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(title)
train_set %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  pull(title)

# Look at the effectiveness of the results
predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
RMSE(predicted_ratings, test_set$rating) # this isn't better than the earlier appoaches, but should be

# Use cross-validation to choose the best lamda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(
  rmse_results,
  tibble(method="Regularized Movie + User Effect Model",  
         RMSE = min(rmses)))
knitr::kable(rmse_results, "html") %>%
  kableExtra::kable_styling(bootstrap_options = "striped", full_width = FALSE)

# ---------------------------------------------------------------------------------------------------------------
#  STEP 5c: Matrix Factorization using recosystem
#  Adapted from https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
# ---------------------------------------------------------------------------------------------------------------

set.seed(123) # This is a randomized algorithm

# TODO - understand whether index1 should be TRUE or FALSE or if it matters
# create the datasets in the right format for recosystem
train_set_new = data_memory(train_set$userId, train_set$movieId, train_set$rating, index1 = TRUE)
test_set_new = data_memory(test_set$userId, test_set$movieId, test_set$rating, index1 = TRUE)

# create the model object
r = Reco()

# use the tune function to find the best tuning parameters
# WARNING! THIS TAKES AWHILE TO RUN, ~ 30 MINUTES
opts = r$tune(train_set_new, opts = list(
  dim = c(10, 20, 30), # number of latent factors
  lrate = c(0.1, 0.2), # learning rate, which can be thought of as the step size in gradient descent
  costp_l1 = 0, # L1 regularization cost for user factors
  costq_l1 = 0, # L1 regulariation cost for item factors
  nthread = 5, # number of threads for parallel computing, the higher this number the more my machine works
  niter = 10 # number of iterations
  ))
opts

# train the model using the best tuning parameters
r$train(train_set_new, opts = c(opts$min, nthread = 1, niter = 20))

# calculate the predicted values
pred_rvec = r$predict(test_set_new, out_memory())
head(pred_rvec, 10)

# Add to the table 
model_matrix_factorization <- RMSE(test_set$rating, pred_rvec)
rmse_results <- bind_rows(
  rmse_results, tibble(method="Matrix Factorization (recosystem)", 
                       RMSE = model_matrix_factorization))

# ---------------------------------------------------------------------------------------------------------------
# STEP 6: Final assessment using the validation data set
# ---------------------------------------------------------------------------------------------------------------

# format the validation data for recosystem, only need userId and movieId since we're predicting rating
validation_set_new = data_memory(validation$userId, validation$movieId, index1 = TRUE)

# calculate the predicted values for rating
pred_rvec = r$predict(validation_set_new, out_memory())
head(pred_rvec, 10)

# Add to the table 
FINAL_model_matrix_factorization <- RMSE(validation$rating, pred_rvec)
rmse_results <- bind_rows(
  rmse_results, tibble(method="FINAL Matrix Factorization (recosystem)", 
                       RMSE = FINAL_model_matrix_factorization))

