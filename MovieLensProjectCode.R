
## Movie Recommendation Engine Project **

#########################################
# Create edx and final_holdout_test sets 
#########################################

#install latex
tinytex::install_tinytex()


#0. Initial production of edx (training set) and final_holdout_test (test set) provided by the course

# Install tidyverse and caret if required, and load

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#save processed sets to rdas folder
#save(edx, file='rdas/edx.rda')
#save(final_holdout_test, file='rdas/final_holdout_test.rda')


#############################
# Initial Data Exploration
#############################


library(dplyr)




#Take a look at datasets
nrow(edx)
ncol(edx)
head(edx)


#Produce list of variables in the edx dataset (and test set), with type and sample entries
edx_rnd <- edx %>% slice_sample(n=10) 
tmp <- capture.output(str(edx_rnd))
tmp2 <- data.frame(tmp[2:length(tmp)])
colnames(tmp2) <- c("AllInfo")
tmp3 <- tmp2 %>% separate(col="AllInfo",into=c("v1","v2"),sep=":") %>% 
  mutate(Variable = substr(v1,3,nchar(v1)),
         Type = substr(v2,2,4),
         Examples = substr(v2,6,100)) %>%
  select(Variable,Type, Examples)
tmp3 %>% knitr::kable()

#Just look at Id 1 so can see quickly load data in the environment window
edx_small <- edx %>% filter(userId == 1)

#Counts of each rating level
cntsbyrating <- edx %>% group_by(rating) %>% summarize(n = n())
cntsbyrating
sum(cntsbyrating$n) #check not lost any

#count distinct user and movies
edx %>% summarize(users = n_distinct(userId),
                  movies = n_distinct(movieId))

#count reviews per genre
cntsbygenre <- edx %>% 
  mutate(Drama_bin = grepl("Drama", genres, fixed = TRUE),
         Comedy_bin = grepl("Comedy", genres, fixed = TRUE),
         Thriller_bin = grepl("Thriller", genres, fixed = TRUE),
         Romance_bin = grepl("Romance", genres, fixed = TRUE)
  )
sum(cntsbygenre$Drama_bin)
sum(cntsbygenre$Comedy_bin)
sum(cntsbygenre$Thriller_bin)
sum(cntsbygenre$Romance_bin)

#Get review counts for some chosen films
cntsfilms <- edx %>% mutate(Frst = as.numeric(grepl("Forrest Gump", title, fixed = TRUE)),
                            Jrp = as.numeric(grepl("Jurassic Park", title, fixed = TRUE)),
                            Plp = as.numeric(grepl("Pulp Fiction", title, fixed = TRUE)),
                            Shr = as.numeric(grepl("The Shawshank Redemption", title, fixed = TRUE)),
                            Spe = as.numeric(grepl("Speed 2: Cruise Control", title, fixed = TRUE))) %>%
  mutate(chosenpic = Frst + Jrp + Plp + Shr + Spe) %>% filter(chosenpic == 1) %>% group_by(title) %>% summarize(n=n())
cntsfilms

#Get rating counts ordered by vol
cntsbyrating2 <- arrange(cntsbyrating,desc(n))
cntsbyrating2

#List the most reviewed movies


#List the least reviewed movies

#Get total number of genres, and counts of each. Order by most reviewed


#sample table for a few users, showing whether they've rated each of the top 5 movies.
keep <- edx %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
tab <- edx %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()


#Produce example matrix of 100 random users and which films reviewed, to see coverage
users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")


#Highest ranked films
#Highest ranked genres


#summary of the data
summary(edx)

tmp2 <- dput(edx_rnd)



#############################
# Data Analysis
#############################



#############################
# Training ML algos
#############################




#############################
# Testing best ML algo
#############################



print(nrow(edx))
cat("Number of rows in training set edx =",nrow(edx))
print(paste0("Current working dir: ", wd))



