
## Movie Recommendation Engine Project **

#########################################
# Create edx and final_holdout_test sets 
#########################################

#install latex
tinytex::install_tinytex()
install.packages("float")

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

#Load libraries
###############

library(dplyr)
library(scales)
library(stringr)
library(tidyverse)
library(caret)
library(lubridate)
library(ggplot2)
library(ggthemes)
#library(kableExtra)
library(float)

# Get Basic Data Info for train/test initial sets
#################################################

#Rows, column frequencies
format(nrow(edx), big.mark = ",", scientific = FALSE)
format(nrow(final_holdout_test), big.mark = ",", scientific = FALSE)
ncol(edx)
head(edx)

#Table to present variables in the edx dataset (and test set), with type and sample entries
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

#Id 1 so can see quickly load data in the environment window
edx_small <- edx %>% filter(userId == 1)


#Data Processing for Charts/Modelling
#####################################

#split title into title and year; add date field for the review date
edx_pr <- edx %>% mutate(filmYear = str_sub(title,-6,-1),
                         title = str_sub(title,end=-7)) %>% 
  mutate(filmYear = gsub('[()]','',filmYear),
         ratingdate = as_datetime(timestamp)) %>%
  mutate(filmDecade = paste0(floor(as.integer(filmYear)/10)*10,'s'),
         Genres_Cnt = 1 + str_count(genres,"\\|")) %>%
  select(-timestamp)

#Split genres column out into separate columns
edx_gnrs <- edx_pr %>% group_by(genres) %>% summarize(n=n()) %>%
  separate(col=genres,into=c("G1","G2","G3","G4","G5","G6","G7","G8"), sep='\\|',fill="right") %>% select(-n)

#Get unique list of all the separate genres
edx_gnrs_vect<- as.vector(as.matrix(edx_gnrs))
genrelist <- unique(edx_gnrs_vect[!is.na(edx_gnrs_vect)])    #remove NAs
genrelist <- genrelist[genrelist != "(no genres listed)"]   #remove where no genres listed
genrelist

#Df of genres in data for presentation
gnre_df <- as.data.frame(genrelist) 
gnre_df

#Add binary variables for each genre. Loops through genres in the genre list and adds a variable with TRUE/FALSE depending
#on whether the specific genre is listed in the genres variable
m <- length(genrelist)
for(n in 1:m){
  newcol <- genrelist[n]
  edx_pr[newcol] <- grepl(newcol, edx_pr$genres, fixed = TRUE)
}

edx_pr_small <- edx_pr %>% filter(userId == 1)

rm(edx) #remove unprocessed edx set for disk mgmt

# Summarize Users vols, movie vols, coverage
############################################

#count distinct user and movies
tmp_usrmov <- edx_pr %>% summarize(users = n_distinct(userId),
                  movies = n_distinct(movieId))
format(tmp_usrmov, big.mark = ",", scientific = FALSE)

#Total ratings if every user had rated every movie
tmp_usrmov_tot <- tmp_usrmov %>% mutate(Combinations = users * movies) %>% select(Combinations)
format(tmp_usrmov_tot, big.mark = ",", scientific = FALSE)

#Percentage coverage
tmp_usrmov_pct <- nrow(edx_pr) / tmp_usrmov_tot$Combinations
percent(tmp_usrmov_pct, accuracy = 0.01)

#Matrix of 100 random users and 100 random films they've reviewed, to visualise coverage
########################################################################################
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")
library(plotly) 

if(!require(webshot)) install.packages("webshot", repos = "http://cran.us.r-project.org")
webshot::install_phantomjs()

set.seed(1986, sample.kind="Rounding")         #set seed for random selection of users/movies

#create matrix to plot
users <- sample(unique(edx_pr$userId), 100)
reshape<- edx_pr %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.)

#Initial plotly chart
fig <- plot_ly(z = reshape, x=1:100, y=1:100, type = "heatmap",colorscale = "Bluered",colorbar = list(title = "Star Rating")) %>% 
  layout(title = "100 random users' Star Ratings for 100 movies", xaxis = list(title = "Movies",showgrid=FALSE,zeroline=FALSE),
         yaxis=list(title="Users",showgrid=FALSE,zeroline=FALSE) )

#Add horizontal line to plot as shapes - offset lines by 0.5 to frame the plotted square
hlines <- function(y = 0, color = "gray") {
  list(type = "line",
       x0 = 0, x1 = 1, xref = "paper", y0 = y+0.5, y1 = y+0.5, 
       line = list(color = color, width = 0.5))
}
#Repeat above 101 times for all the horizontal lines
fig$x$layout$shapes <- lapply(0:100, hlines)


#Add vertical line to plot as shapes - offset lines by 0.5 to frame the plotted square
vlines <- function(x = 0, color = "gray") {
  list(type = "line",
       y0 = 0, y1 = 1, yref = "paper", x0 = x+0.5, x1 = x+0.5,
       line = list(color = color, width = 0.5))
}
#Repeat above 101 times for all the vertical lines
fig$x$layout$shapes[101:201] <- lapply(0:100, vlines)   #Adding 101 new shapes for each vertical line, in addition to horizontal ones

#show plot, NB: expanded size of window by 1.4x in pdf report, and reduced line width of horizontal/vertical lines
fig

#save plot as png
p <- fig
file = "fig.png"; format = "png"
debug=verbose=safe=F
b <- plotly_build(p)
plotlyjs <- plotly:::plotlyjsBundle(b)
plotlyjs_path <- file.path(plotlyjs$src$file, plotlyjs$script)
if (!is.null(plotlyjs$package)) {
  plotlyjs_path <- system.file(plotlyjs_path, package = plotlyjs$package)
}
tmp <- tempfile(fileext = ".json")
cat(plotly:::to_JSON(b$x[c("data", "layout")]), file = tmp)
args <- c("graph", tmp, "-o", file, "--format", 
          format, "--plotlyjs", plotlyjs_path, if (debug) "--debug", 
          if (verbose) "--verbose", if (safe) "--safe-mode")
base::system(paste("orca", paste(args, collapse = " ")))

#print png image of plot
knitr::include_graphics("fig.png")

#Movie Analysis
##################

#Histograms of rating volumes per movie (so most movies with v few ratings)
edx_pr %>% group_by(movieId) %>% summarize(n=n()) %>% arrange(., desc(n)) %>% #filter(n>100) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=50, fill="steelblue") +
  ylab("Number of Movies") +
  xlab("Number of Ratings") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

edx_pr %>% group_by(movieId) %>% summarize(n=n()) %>% arrange(., desc(n)) %>% #filter(n>100) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=200, fill="steelblue") + 
  xlim(1,200) +
  ylab("Number of Movies") +
  xlab("Number of Ratings") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

#Top 10 movies by volume of ratings 
edx_pr %>% select(title) %>% group_by(title) %>% 
  summarize(ratingsVol = n()) %>% 
  arrange(desc(ratingsVol)) %>% .[1:10,] %>%
  mutate(title = substr(title,1,35)) %>% knitr::kable()

#Bottom 10 movies by volume of ratings 
edx_pr %>% select(title) %>% group_by(title) %>% 
  summarize(ratingsVol = n()) %>% 
  arrange(ratingsVol) %>% .[1:10,] %>% 
  mutate(title = substr(title,1,35)) %>% knitr::kable()


#Distribution of Average Ratings per Movie Chart
#(Showing some movies tend to rank higher/lower on average)
edx_pr %>% group_by(movieId) %>% summarize(averageRating = mean(rating)) %>% 
  ggplot(aes(averageRating)) +
  geom_histogram(bins = 30, fill = "steelblue") +
  xlab("Average Rating") + 
  ylab("Number of Movies") +
  theme_economist(horizontal=TRUE) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))  

#Top 10 rated movies (films with minimum 100 ratings) 
edx_pr %>% select(title,rating) %>% group_by(title) %>% 
  summarize(ratingsVol = n(), Avgrating = mean(rating)) %>% 
  mutate(Avgrating = round(Avgrating,3)) %>%
  filter(ratingsVol >= 100) %>%
  select(-ratingsVol) %>%
  arrange(desc(Avgrating)) %>%  .[1:10,] %>% knitr::kable()

#Bottom 10 rated movies (films with minimum 100 ratings) 
edx_pr %>% select(title,rating) %>% group_by(title) %>% 
  summarize(ratingsVol = n(), Avgrating = mean(rating)) %>% 
  mutate(Avgrating = round(Avgrating,3)) %>%
  filter(ratingsVol >= 100) %>%
  select(-ratingsVol) %>%
  arrange(Avgrating) %>%  .[1:10,] %>% knitr::kable()

#Mean and Median vol of ratings per movie
ratingsPerMovie <- edx_pr %>% select(movieId) %>% group_by(movieId) %>% summarize(n=n()) %>% 
  summarize(Mean_RatingsPerMovie = mean(n),Median_RatingsPerMovie = median(n)) %>% as.data.frame() %>% 
  mutate(Mean_RatingsPerMovie = round(Mean_RatingsPerMovie,1))
ratingsPerMovie



#Movie Genre Analysis

#Overall number of genre combinations:
tmp_gnre <- edx_pr %>% summarize(genres = n_distinct(genres))
tmp_gnre

#Top 10 rated genre combinations (min 1000 ratings)
edx_pr %>% select(genres, rating) %>% group_by(genres) %>% summarize(AvgRating = mean(rating),cnt=n()) %>%
  filter(cnt > 1000) %>%  select(-cnt) %>% arrange(desc(AvgRating)) %>% .[1:10,] %>% 
  mutate(genres = substr(genres,1,35),AvgRating = round(AvgRating,3)) %>% knitr::kable()

#Bottom 10 rated genre combinations (min 1000 ratings)
edx_pr %>% select(genres, rating) %>% group_by(genres) %>% summarize(AvgRating = mean(rating),cnt=n()) %>%
  filter(cnt > 1000) %>%  select(-cnt) %>% arrange(AvgRating) %>% .[1:10,] %>% 
  mutate(genres = substr(genres,1,35),AvgRating = round(AvgRating,3)) %>% knitr::kable()


#Produce summary table of average rating and count per genre, then plot (nb, m= number of genres)
for(n in 1:m){
  col <- genrelist[n]
  tmp <- edx_pr %>% group_by_at(col) %>% summarize(avgRating = mean(rating), RatingsCount=n()) %>% mutate(Genre = col)
  names(tmp)[names(tmp) == col] <- 'TrueOrFalse'
  tmp <- tmp %>% filter(TrueOrFalse == TRUE) %>% select(-TrueOrFalse)
  if (n==1){
    genre_df <- tmp
  } else {
    genre_df <- rbind(genre_df,tmp)
  }
}

genre_df %>%
  ggplot(aes(group=1)) +
  geom_bar(aes(reorder(Genre,-avgRating,sum), RatingsCount),stat="identity",fill="steelblue") +
  geom_line(aes(Genre,avgRating/0.000001),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.000001, name="Average Rating")) +
  ylab("Volume of Ratings") +
  xlab("Genres") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.text.x = element_text(angle=90,hjust=1),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10)))


#Plot distribution of ratings by number of genres listed for the film
edx_pr %>% group_by(Genres_Cnt) %>%
  summarize(avgRating = mean(rating), n=n()) %>%
  ggplot() +
  geom_bar(aes(Genres_Cnt,n),stat="identity",fill="steelblue") +
  geom_line(aes(Genres_Cnt,avgRating/0.000002),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.000002, name="Average Rating")) +
  scale_x_continuous(breaks = 1:8) +
  ylab("Volume of Ratings") +
  xlab("Number of genres") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10))) 


#Movie Year Analysis

#Function to help neaten x axis
every_nth = function(n) {
  return(function(x) {x[c(TRUE, rep(FALSE, n - 1))]})
}

#Plot volume of ratings by film Year
edx_pr %>% select(filmYear) %>% group_by(filmYear) %>% summarize(n=n()) %>% 
  ggplot(aes(x=filmYear,y=n)) +
  geom_bar(stat="identity", fill="steelblue") +
  scale_y_continuous(labels = scales::label_number_si()) +
  scale_x_discrete(breaks = every_nth(n = 5)) +
  xlab("Film Release Year") + 
  ylab("Number of Ratings") +
  theme_economist(horizontal=TRUE) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Plot volume of ratings by film Decade
edx_pr %>% group_by(filmDecade) %>% summarize(n=n()) %>% 
  ggplot(aes(x=filmDecade,y=n)) +
  geom_bar(stat="identity", fill="steelblue") +
  scale_y_continuous(labels = scales::label_number_si()) +
  xlab("Film Release Decade") + 
  ylab("Number of Ratings") +
  theme_economist(horizontal=FALSE) +
  coord_flip() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Boxplot of ratings volume per film per year. Shows some films getting way more than average, unsurprisingly
edx_pr %>% group_by(movieId) %>% summarize(filmYear = mean(as.integer(filmYear)), numratings = n()) %>% 
  arrange(filmYear) %>% mutate(filmYear = as.factor(filmYear)) %>%
  ggplot(aes(filmYear, numratings)) +
  geom_boxplot() + 
  scale_x_discrete(breaks = every_nth(n = 5)) +
  theme(axis.text.x = element_text(angle=90,hjust=1)) +
  scale_y_continuous(trans="sqrt") +
  xlab("Year of Film Release") + 
  ylab("Number of Ratings (sqrt transformed)") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

#Plot film year vs average rating
edx_pr %>% select(filmYear,rating) %>% mutate(filmYear = as.integer(filmYear)) %>%
  group_by(filmYear) %>%
  summarize(avgRating = mean(rating)) %>% 
  ggplot(aes(filmYear,avgRating)) +
  geom_point() +
  geom_smooth() +
  scale_x_continuous(breaks = seq(1915,2015,by=10)) +
  xlab("Year of Film Release") + 
  ylab("Average Film Rating") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 


#Rating vs Number of reviews per Movie
tmp <- edx_pr %>% select(movieId, rating) %>% group_by(movieId) %>% summarize(ratingVol=n(), AvgRating = mean(rating)) %>% 
  mutate(ratingVolGrp_tmp = cut(ratingVol, breaks=c(0,25, 50, 100,250,500,1000,2500, 5000,Inf),dig.lab=4)) %>% group_by(ratingVolGrp_tmp) %>%
  summarize(Ratings = n(), Mean_Rating = mean(AvgRating)) %>% 
  mutate(ratingVolGrp = str_replace_all(ratingVolGrp_tmp,"([\\(\\]])","")) %>%
  mutate(ratingVolGrp = str_replace_all(ratingVolGrp,"([\\,])","-")) 
tmp$ratingVolGrp <- factor(tmp$ratingVolGrp, levels = tmp$ratingVolGrp)  #lock in ordering for x-axis in plot
tmp %>%
  ggplot() +
  geom_bar(aes(ratingVolGrp,Ratings),stat="identity",fill="steelblue") +
  geom_line(aes(ratingVolGrp,Mean_Rating/0.002,group=1),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(accuracy = 0.1), sec.axis=sec_axis(~.*0.002, name="Average Rating")) +
  #  scale_x_discrete(labels=scales::label_number_si()) +
  # scale_x_continuous(breaks = 1:8) +
  ylab("Volume of Ratings") +
  xlab("Movie Rating Volumes") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10))) 


#User Analysis
####################

#Histogram of rating volumes per users. So mode number of reviews is about 25 per user
edx_pr %>% group_by(userId) %>% summarize(n=n()) %>% arrange(., desc(n)) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins=100, fill="steelblue") +
  ylab("Number of Users") +
  xlab("Number of Ratings") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

#Same histogram, but focusing in on reviewers with up to 200 reviews
edx_pr %>% group_by(userId) %>% summarize(n=n()) %>% arrange(., desc(n)) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins=100, fill="steelblue") +
  xlim(1,200) +
  ylab("Number of Users") +
  xlab("Number of Ratings") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

#Top 10 users by volume of ratings 
edx_pr %>% select(userId) %>% group_by(userId) %>% 
  summarize(ratingsVol = n()) %>% 
  arrange(desc(ratingsVol)) %>% .[1:10,] %>% knitr::kable()

#Bottom 10 users by volume of ratings 
edx_pr %>% select(userId) %>% group_by(userId) %>% 
  summarize(ratingsVol = n()) %>% 
  arrange(ratingsVol) %>% .[1:10,] %>% knitr::kable()

#Mean and Median vol of ratings per user
ratingsPerUser <- edx_pr %>% select(userId) %>% group_by(userId) %>% summarize(n=n()) %>% 
  summarize(Mean_RatingsPerUser = mean(n),Median_RatingsPerUser = median(n)) %>% as.data.frame() %>% 
  mutate(Mean_RatingsPerUser = round(Mean_RatingsPerUser,1))
ratingsPerUser

#Distribution of Average Ratings per User Chart
  #(Showing some users tend to rank higher/lower on average)
edx_pr %>% group_by(userId) %>% summarize(averageRating = mean(rating)) %>% 
  ggplot(aes(averageRating)) +
    geom_histogram(bins = 30, fill = "steelblue") + 
  xlab("Average Rating") + 
  ylab("Number of Users") +
    theme_economist(horizontal=TRUE) +
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
          axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))

#Plot week of review vs average rating
edx_pr %>% select(ratingdate, rating) %>% mutate(ratingWeek = round_date(ratingdate,unit="week")) %>% 
  group_by(ratingWeek) %>% summarize(avgRating = mean(rating)) %>% 
  ggplot(aes(ratingWeek,avgRating)) +
  geom_point() +
  geom_smooth() +
  scale_x_datetime(date_breaks = "1 year",date_labels = "%Y") +
  xlab("Rating Week") + 
  ylab("Average Film Rating") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Hour of review in the day. Most ratings done in evening, but little impact on average ratings 
edx_pr %>% select(ratingdate,rating) %>% mutate(ratingTime = hour(ratingdate)) %>% group_by(ratingTime) %>%
  summarize(avgRating = mean(rating),n=n()) %>% 
  ggplot() +
  geom_bar(aes(ratingTime,n),stat="identity",fill="steelblue") +
  geom_line(aes(ratingTime,avgRating/0.00001),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.00001, name="Average Rating")) +
  #scale_x_continuous(breaks = 1:8) +
  ylab("Volume of Ratings") +
  xlab("Hour of Day") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10))) 

#Days since user's first review 
usr_dt <- edx_pr %>% select(c('userId','ratingdate')) %>% group_by(userId) %>% summarize(min_UsrRatingDate = min(ratingdate)) 
edx_pr %>% select(c('userId','ratingdate','rating')) %>% left_join(usr_dt, by='userId') %>% 
  mutate(DaysFromUserFirstReview = as.numeric(as.Date(ratingdate) - as.Date(min_UsrRatingDate))) %>% 
  group_by(DaysFromUserFirstReview) %>% summarize(avgRating = mean(rating),n=n()) %>% 
  ggplot(aes(DaysFromUserFirstReview,avgRating)) +
  # geom_bar(aes(DaysFromUserFirstReview,n),stat="identity",fill="steelblue") +
  geom_point(color="steelblue") +
  geom_smooth() +
  scale_y_continuous(labels = scales::label_number_si()) +
  #scale_x_continuous(breaks = 1:8) +
  ylab("Average Rating") +
  xlab("Days since first user review") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Days since any user rated a given movie
mov_dt <- edx_pr %>% select(c('movieId','ratingdate')) %>% group_by(movieId) %>% summarize(min_MovRatingDate = min(ratingdate)) 
edx_pr %>% select(c('movieId','ratingdate','rating')) %>% left_join(mov_dt, by='movieId') %>% 
  mutate(DaysFromMovieFirstReview = as.numeric(as.Date(ratingdate) - as.Date(min_MovRatingDate))) %>% 
  group_by(DaysFromMovieFirstReview) %>% summarize(avgRating = mean(rating),n=n()) %>% 
  ggplot(aes(DaysFromMovieFirstReview,avgRating)) +
  # geom_bar(aes(DaysFromUserFirstReview,n),stat="identity",fill="steelblue") +
  geom_point(color="steelblue") +
  geom_smooth() +
  scale_y_continuous(labels = scales::label_number_si()) +
  #scale_x_continuous(breaks = 1:8) +
  ylab("Average Rating") +
  xlab("Days since first movie review") +
  ylim(3,4) +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 
  
#Rating vs Number of reviews per User
tmp <- edx_pr %>% select(userId, rating) %>% group_by(userId) %>% summarize(ratingVol=n(), AvgRating = mean(rating)) %>% 
  mutate(ratingVolGrp_tmp = cut(ratingVol, breaks=c(0,25,50,100,150,200,250,500,1000,Inf),dig.lab=4)) %>% group_by(ratingVolGrp_tmp) %>%
summarize(Ratings = n(), Mean_Rating = mean(AvgRating)) %>% 
  mutate(ratingVolGrp = str_replace_all(ratingVolGrp_tmp,"([\\(\\]])","")) %>%
  mutate(ratingVolGrp = str_replace_all(ratingVolGrp,"([\\,])","-")) 
tmp$ratingVolGrp <- factor(tmp$ratingVolGrp, levels = tmp$ratingVolGrp)  #lock in ordering for x-axis in plot
tmp %>%
  ggplot() +
  geom_bar(aes(ratingVolGrp,Ratings),stat="identity",fill="steelblue") +
  geom_line(aes(ratingVolGrp,Mean_Rating/0.00025,group=1),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.00025, name="Average Rating")) +
#  scale_x_discrete(labels=scales::label_number_si()) +
 # scale_x_continuous(breaks = 1:8) +
  ylab("Volume of Ratings") +
  xlab("User Rating Volumes") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10))) 


#Preferences for specific User
###############################

#Choose a user with over 500 reviews at random
set.seed(1535, sample.kind="Rounding")
SuperUser <- edx_pr %>% select(userId) %>% group_by(userId) %>% summarize(n=n()) %>%
  filter(n >= 500) %>% slice_sample(n=1) %>% .$userId

SuperUserdf <- edx_pr %>% filter(userId == SuperUser) 

#Genre ratings for randomly chosen super user 
for(n in 1:m){
  col <- genrelist[n]
  tmp <- SuperUserdf %>% group_by_at(col) %>% summarize(avgRating = mean(rating), RatingsCount=n()) %>% mutate(Genre = col)
  names(tmp)[names(tmp) == col] <- 'TrueOrFalse'
  tmp <- tmp %>% filter(TrueOrFalse == TRUE) %>% select(-TrueOrFalse)
  if (n==1){
    genre_df_su <- tmp
  } else {
    genre_df_su <- rbind(genre_df_su,tmp)
  }
}

genre_df_su %>%
  ggplot(aes(group=1)) +
  geom_bar(aes(reorder(Genre,-avgRating,sum), RatingsCount),stat="identity",fill="steelblue") +
  geom_line(aes(Genre,avgRating/0.01),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.01, name="Average Rating")) +
  ylab("Volume of Ratings") +
  xlab("Genres") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.text.x = element_text(angle=90,hjust=1),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10)))

#Top 40 rated movies for same user 
SuperUserdf %>% select(title,rating) %>% group_by(title) %>% 
  summarize(ratingsVol = n(), Avgrating = mean(rating)) %>% 
  mutate(Avgrating = round(Avgrating,3)) %>%
  select(-ratingsVol) %>%
  arrange(desc(Avgrating)) %>%  .[1:40,] %>% knitr::kable()

#Bottom 40 rated movies for same user 
SuperUserdf %>% select(title,rating) %>% group_by(title) %>% 
  summarize(ratingsVol = n(), Avgrating = mean(rating)) %>% 
  mutate(Avgrating = round(Avgrating,3)) %>%
  select(-ratingsVol) %>%
  arrange(Avgrating) %>%  .[1:40,] %>% knitr::kable()

#Movie year ratings for super user
SuperUserdf %>% select(filmYear,rating) %>% mutate(filmYear = as.integer(filmYear)) %>%
  group_by(filmYear) %>%
  summarize(avgRating = mean(rating)) %>% 
  ggplot(aes(filmYear,avgRating)) +
  geom_point() +
  geom_smooth() +
  scale_x_continuous(breaks = seq(1915,2015,by=10)) +
  xlab("Year of Film Release") + 
  ylab("Average Film Rating") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 



##################
# Training models
##################

#Ultimate goal is to minimise RMSE in the test set. RMSE function is below comparing predicted vs actual ratings
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Experimenting with models in full training set
################################################

#Model 1
########
mu <- mean(edx_pr$rating)   
edx_pr <- edx_pr %>% mutate(pred1 = mu)  

#Model 2: Average for all films + movie effect
########
movie_avgs <- edx_pr %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu)) #average difference in rating for a given movie vs all movies
edx_pr <- edx_pr %>% left_join(movie_avgs, by='movieId') %>%
  mutate(pred2 = mu + b_i) #add b_i and 2nd model prediction 

#Model 3: Average for all films + movie effect + user effect
#########
user_avgs <- edx_pr %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) #average difference in rating for a given user 
edx_pr <- edx_pr %>% left_join(user_avgs, by='userId') %>%
  mutate(pred3 = mu + b_i + b_u) #add b_i and 3rd model prediction 


#Do some residuals plots; should be white noise if capturing all signal

#Movie Year
edx_pr %>% mutate(resid = pred3 - rating, filmYear = as.integer(filmYear)) %>% group_by(filmYear) %>% 
  summarize(resid = mean(resid)) %>%
  ggplot(aes(filmYear,resid)) +
  geom_point() +
  geom_smooth() +
  scale_x_continuous(breaks = seq(1910,2010,by=10)) +
  xlab('Movie Year') + 
  ylab('Average Residual') +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Movie Decade
edx_pr %>% mutate(resid = pred3 - rating) %>% group_by(filmDecade ) %>% summarize(resid = mean(resid)) %>%
  ggplot(aes(filmDecade,resid)) +
  geom_point() +
  geom_smooth() +
  xlab('Movie Decade') + 
  ylab('Average Residual') +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Genre count
edx_pr %>% mutate(resid = pred3 - rating) %>% group_by(Genres_Cnt ) %>% summarize(resid = mean(resid)) %>%
  ggplot(aes(Genres_Cnt ,resid)) +
  geom_point() +
  geom_smooth() +
  # scale_x_continuous(breaks = seq(1910,2010,by=10)) +
  xlab('Genre Count') + 
  ylab('Average Residual') +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Genres (m defined above as elements in genrelist)
for(n in 1:m){
  col <- genrelist[n]
  tmp <- edx_pr %>% mutate(resid = pred3 - rating) %>% group_by_at(col) %>% summarize(resid = mean(resid), RatingsCount=n()) %>% mutate(Genre = col)
  names(tmp)[names(tmp) == col] <- 'TrueOrFalse'
  tmp <- tmp %>% filter(TrueOrFalse == TRUE) %>% select(-TrueOrFalse)
  #tmp <- edx_pr %>% group_by_at(col) %>% summarize(AvgRating = mean(rating), RatingsCount=n()) %>% mutate(Genre = col)
  if (n==1){
    genre_df2 <- tmp
  } else {
    genre_df2 <- rbind(genre_df2,tmp)
  }
}

genre_df2 %>%
  ggplot(aes(group=1)) +    #group=1 required to overlay line on bar chart
  geom_bar(aes(reorder(Genre,-resid,sum), RatingsCount),stat="identity",fill="steelblue") +
  geom_line(aes(Genre,resid/0.000001),color="red",size=1.5) +
  scale_y_continuous(labels = scales::label_number_si(), sec.axis=sec_axis(~.*0.000001, name="Average Redisual")) +
  ylab("Volume of Ratings") +
  xlab("Genres") +
  theme_economist() +
  theme(axis.text.y.left=element_text(color="steelblue"),
        axis.text.y.right=element_text(color="red"),
        axis.text.x = element_text(angle=90,hjust=1),
        axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10)))


#NB: On reflection, no sure there's any real point using genre as an additional variable unless it's adjusted for each 
#specific user. You've already got a movie specific effect, and genre just a higher level categorisation of the movies.
#So overall impact of genre across all users shouldn't really add anything beyond the specific movie impact

#Genre preference at user level could be an avenue though - i.e. adjusting the user average ratings by each user's 
#genre preference

#Likewise, year of movie should really already be captured by the film, although residuals plot suggests maybe
#it's not all captured.
#Again, if could capture each user's preference for new vs old movies that might be something additional, but the 
#overall impact of movie year ought to be captured by having the overall rating of each movie

#Try 1) days since movie first rated, 2) days since user first rated, 3) number of people who've rated the movie

#Days since movie's first rating (maybe something around v first reviewers before settling down)
edx_pr %>% mutate(resid = pred3 - rating) %>% select(c('movieId','ratingdate','resid')) %>% 
  left_join(mov_dt, by='movieId') %>% 
  mutate(DaysFromMovieFirstReview = as.numeric(as.Date(ratingdate) - as.Date(min_MovRatingDate))) %>% 
  group_by(DaysFromMovieFirstReview) %>% summarize(resid = mean(resid),n=n()) %>% 
  ggplot(aes(DaysFromMovieFirstReview,resid)) +
  # geom_bar(aes(DaysFromMovieFirstReview,n),stat="identity",fill="steelblue") +
  geom_point(color="steelblue") +
  geom_smooth() +
  scale_y_continuous(labels = scales::label_number_si()) +
  #scale_x_continuous(breaks = 1:8) +
  ylab("Average Residual") +
  xlab("Days since first movie review") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 

#Days since user's first rating (not much here)
edx_pr %>% mutate(resid = pred3 - rating) %>% select(c('userId','ratingdate','resid')) %>% 
  left_join(usr_dt, by='userId') %>% 
  mutate(DaysFromUserFirstReview = as.numeric(as.Date(ratingdate) - as.Date(min_UsrRatingDate))) %>% 
  group_by(DaysFromUserFirstReview) %>% summarize(resid = mean(resid),n=n()) %>% 
  ggplot(aes(DaysFromUserFirstReview,resid)) +
  # geom_bar(aes(DaysFromUserFirstReview,n),stat="identity",fill="steelblue") +
  geom_point(color="steelblue") +
  geom_smooth() +
  scale_y_continuous(limits=c(-1,1)) +
  #scale_x_continuous(breaks = 1:8) +
  ylab("Average Residual") +
  xlab("Days since first user review") +
  theme_economist() +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0))) 


#Model 4: Average for all films + movie effect + user effect, regularized (lambda=5, see below)
#########
lambda <- 5  
movie_avgs_reg <- edx_pr %>%
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu)/(n()+lambda))
user_avgs_reg <- edx_pr %>% 
  left_join(movie_avgs_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+lambda))
edx_pr <- edx_pr %>% 
  left_join(movie_avgs_reg, by = "movieId") %>%
  left_join(user_avgs_reg, by = "userId") %>%
  mutate(pred4 = mu + b_i_reg + b_u_reg) #add b_i_reg and 3rd model prediction into validation set


#Finding best lambda for user/movie model regularization; 
#NB: just using single validation set for runtime reasons

valid_index <- grp_index[["Fold04"]]
Lam_valid_set <- edx_pr[tmp_valid_index,]
Lam_train_set <- edx_pr[-tmp_valid_index,]

#Remove any users/movies in validation that aren't in train
Lam_valid_set <- Lam_valid_set %>% 
  semi_join(Lam_train_set, by = "movieId") %>%
  semi_join(Lam_train_set, by = "userId") 

#Find best lambda for movie and user model
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(Lam_train_set$rating)
  b_i <- Lam_train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- Lam_train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    Lam_valid_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, Lam_valid_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda  #Best lambda is 5

rm("Lam_train_set","Lam_valid_set") # remove Lam_train_set and Lam_valid_set for memory management



#Model 5: Model 3 but adjusted for a user-specific genre preference
#########

MinVol = 5

mu <- mean(edx_pr$rating)  
movie_avgs <- edx_pr %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
user_avgs <- edx_pr %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_gnre_avgs <- user_avgs %>% select(userId)   #Just get user IDs in train set

#Loop through genres and create a set with each genre's bias for each user
for(x in genrelist){
  user_gnre_avgs_tmp <- edx_pr %>% select(movieId,userId,x,rating) %>%
    left_join(movie_avgs, by='movieId') %>%     #to get b_i
    left_join(user_avgs, by='userId') %>%       #to get b_u
    group_by_at(c('userId', x)) %>%   
    summarize(n=n(), b_u_g = mean(rating - mu - b_i - b_u)) %>%
    filter(get(x) == TRUE) %>% filter(n >= MinVol) %>% select(userId, b_u_g) 
  
  names(user_gnre_avgs_tmp)[names(user_gnre_avgs_tmp) == "b_u_g"] <- paste0(x,"_b_u_g") #change b_u_g var name to "Action_b_u_g" etc
  
  user_gnre_avgs <- user_gnre_avgs %>% left_join(user_gnre_avgs_tmp, by='userId') #Join onto the df for all genres
  
}

#replace all NAs with zeroes (fewer than minvol ratings for the genre, so give them zero bias)
user_gnre_avgs <- user_gnre_avgs %>% replace(is.na(.), 0)

#Calc the overall average bias given the genres of each movie for a given user
#NB: Do in chunks given memory issues, and just keep final bias per row

GetBug <- function(a,b){
  df <- edx_pr[a:b,] %>% select(c('userId', 'movieId', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 
                                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')) %>% mutate() %>% 
    left_join(user_gnre_avgs,by='userId') %>% rename(FilmNoir = "Film-Noir", SciFi= "Sci-Fi",
                                                     FilmNoir_b_u_g = "Film-Noir_b_u_g", SciFi_b_u_g= "Sci-Fi_b_u_g") %>%
    mutate(b_u_g_num = Action*Action_b_u_g + Adventure*Adventure_b_u_g + Animation*Animation_b_u_g +
             Children*Children_b_u_g + Comedy*Comedy_b_u_g + Crime*Crime_b_u_g + Documentary*Documentary_b_u_g + 
             Drama*Drama_b_u_g + Fantasy*Fantasy_b_u_g +
             FilmNoir*FilmNoir_b_u_g + Horror*Horror_b_u_g + IMAX*IMAX_b_u_g +
             Musical*Musical_b_u_g + Mystery*Mystery_b_u_g + Romance*Romance_b_u_g +
             SciFi*SciFi_b_u_g + Thriller*Thriller_b_u_g + War*War_b_u_g +Western*Western_b_u_g ,
           b_u_g_denom = rowSums(across(c(Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, 
                                          FilmNoir, Horror, IMAX, Musical, Mystery, Romance, SciFi, Thriller, 
                                          War, Western)))) %>% 
    mutate(b_u_g = b_u_g_num / b_u_g_denom) %>% select(userId, movieId,b_u_g)
  return(df)
}

#Run function for each chunk of rows
slice1 <- GetBug(1,3000000)
slice2 <- GetBug(3000001,6000000)
slice3 <- GetBug(6000001,nrow(edx_pr))

user_gnre_avgs_fin <- rbind(slice1, slice2, slice3)   #Bind the chunks together

rm(slice1, slice2, slice3) #remove the separate chunks for memory mgmt

#Join the genre biases back into the core dataset and reset handful of NAs for one movie without genre to zero bias
edx_pr <- edx_pr %>% left_join(user_gnre_avgs_fin, by=c('userId','movieId')) %>% mutate(b_u_g = ifelse(is.na(b_u_g), 0, b_u_g))

edx_pr_small <- edx_pr %>% filter(userId == 1)

#get predicted using new genre bias
predicted_ratings <- edx_pr %>% mutate(pred5 = pred3 + b_u_g) %>% .$pred5

RMSE(edx_pr$pred3,edx_pr$rating)
RMSE(predicted_ratings,edx_pr$rating)  #Improves things a good bit in training. Check if holds in cross validation.



#Using K-fold cross validation to give better sense for each model's out of sample performance
##############################################################################################

#Chosen k=5; mainly a runtime decision.
#Note, if model fitting using lm would just use trainControl for k-fold, but 
#given method used (and lm runtime extremely slow for this many params) will do k-fold manually

#set k and create index vars with rows assigned to each fold
k <- 5
grp_index <- createFolds(y = edx_pr$rating, k = k, list = TRUE, returnTrain = FALSE)

#Make sure no preds or biases in edx_pr before training models
edx_pr <- edx_pr %>% select(-any_of(c('b_i','b_u','b_u_g','b_i_reg','b_u_reg','pred1','pred2','pred3','pred4','pred5')))

#Loop through each of the k-folds; each time train on 80%, validate on 20%. Then collect average results in validation sets
for(i in 1:k){
  
  #Clear up spare space
  rm(tmp_train)
  rm(tmp_valid)
  gc()
  
  #Get the index for each cross validation valid vs train sets
  if(i < 10 & k >= 10){
    tmp_valid_index <- grp_index[[paste0("Fold0",i)]]
  } else {
    tmp_valid_index <- grp_index[[paste0("Fold",i)]]
  }
  
  #filter to create cross validation valid vs train sets
  tmp_valid <- edx_pr[tmp_valid_index,]
  tmp_train <- edx_pr[-tmp_valid_index,]
  
  #Get rid of any movies or users in the validation set that aren't in the train set
  tmp_valid <- tmp_valid %>% 
    semi_join(tmp_train, by = "movieId") %>%
    semi_join(tmp_train, by = "userId")
  
  #Use training sets for following model builds, then use models to predict in validation set
  
  ###
  
  #Model 1: Average for all films
  mu <- mean(tmp_train$rating)   
  tmp_valid <- tmp_valid %>% mutate(pred1 = mu)  #add into validation set
  
  #Model 2: Average for all films + movie effect
  movie_avgs <- tmp_train %>%
    group_by(movieId) %>%
    summarize(b_i = mean(rating - mu)) #average difference in rating for a given movie vs all movies
  tmp_valid <- tmp_valid %>% left_join(movie_avgs, by='movieId') %>%
    mutate(pred2 = mu + b_i) #add b_i and 2nd model prediction into validation set
  
  #Model 3: Average for all films + movie effect + user effect
  user_avgs <- tmp_train %>%
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i)) #average difference in rating for a given user 
  tmp_valid <- tmp_valid %>% left_join(user_avgs, by='userId') %>%
    mutate(pred3 = mu + b_i + b_u) #add b_i and 3rd model prediction into validation set
    
  #Model 4: Average for all films + movie effect + user effect, regularized (lambda=5, see below)
  lambda <- 5  
  movie_avgs_reg <- tmp_train %>%
      group_by(movieId) %>%
      summarize(b_i_reg = sum(rating - mu)/(n()+lambda))
  user_avgs_reg <- tmp_train %>% 
      left_join(movie_avgs_reg, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+lambda))
  tmp_valid <- tmp_valid %>% 
      left_join(movie_avgs_reg, by = "movieId") %>%
      left_join(user_avgs_reg, by = "userId") %>%
      mutate(pred4 = mu + b_i_reg + b_u_reg) #add b_i_reg and 3rd model prediction into validation set

  
  #Model 5: model 3 + Genre effect per user. Min 5 genre ratings for the average to be used (set to zero if less than 5)
  MinVol = 5
  user_gnre_avgs <- user_avgs %>% select(userId)   #Get user IDs in train set
  
  #Loop through genres and create a set with each genre's bias for each user
  for(x in genrelist){
    user_gnre_avgs_tmp <- tmp_train %>% select(movieId,userId,x,rating) %>%
      left_join(movie_avgs, by='movieId') %>%                             #to get b_i
      left_join(user_avgs, by='userId') %>%                               #to get b_u
      group_by_at(c('userId', x)) %>%                                     #group by user and genre
      summarize(n=n(), b_u_g = mean(rating - mu - b_i - b_u)) %>%         #get bias for the genre
      filter(get(x) == TRUE) %>%                                          #only retain figures for ratings related to the genre x
      filter(n >= MinVol) %>% select(userId, b_u_g)                       #only retain where user gave 5 ratings for genre x 
    
    names(user_gnre_avgs_tmp)[names(user_gnre_avgs_tmp) == "b_u_g"] <- paste0(x,"_b_u_g") #change b_u_g var name to "Action_b_u_g" etc
    
    user_gnre_avgs <- user_gnre_avgs %>% left_join(user_gnre_avgs_tmp, by='userId') #Join onto the df for all genres
    
  }
  
  #replace all NAs with zeroes (fewer than minvol ratings for the genre, so give them zero bias)
  user_gnre_avgs <- user_gnre_avgs %>% replace(is.na(.), 0)
  
  #Calc the overall average bias given the genres of each movie for a given user in the validation set
  user_gnre_avgs_fin <- tmp_valid %>% select(c('userId', 'movieId', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 
                                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')) %>% mutate() %>% 
    left_join(user_gnre_avgs,by='userId') %>% rename(FilmNoir = "Film-Noir", SciFi= "Sci-Fi",
                                                     FilmNoir_b_u_g = "Film-Noir_b_u_g", SciFi_b_u_g= "Sci-Fi_b_u_g") %>%
    mutate(b_u_g_num = Action*Action_b_u_g + Adventure*Adventure_b_u_g + Animation*Animation_b_u_g +
             Children*Children_b_u_g + Comedy*Comedy_b_u_g + Crime*Crime_b_u_g + Documentary*Documentary_b_u_g + 
             Drama*Drama_b_u_g + Fantasy*Fantasy_b_u_g +
             FilmNoir*FilmNoir_b_u_g + Horror*Horror_b_u_g + IMAX*IMAX_b_u_g +
             Musical*Musical_b_u_g + Mystery*Mystery_b_u_g + Romance*Romance_b_u_g +
             SciFi*SciFi_b_u_g + Thriller*Thriller_b_u_g + War*War_b_u_g +Western*Western_b_u_g ,
           b_u_g_denom = rowSums(across(c(Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, 
                                          FilmNoir, Horror, IMAX, Musical, Mystery, Romance, SciFi, Thriller, 
                                          War, Western)))) %>% 
    mutate(b_u_g = b_u_g_num / b_u_g_denom) %>% select(userId, movieId,b_u_g)
  
  
  #Join the genre biases onto the validation dataset and reset handful of NAs for one movie without genre to zero bias
  tmp_valid <- tmp_valid %>% left_join(user_gnre_avgs_fin, by=c('userId','movieId')) %>% 
    mutate(b_u_g = ifelse(is.na(b_u_g), 0, b_u_g)) %>% mutate(pred5 = mu + b_i + b_u + b_u_g)
  
  ###
  
  #Stock the RMSEs for valid set in a var for each model, and stock in df
  rmse1_tmp <- RMSE(tmp_valid$rating, tmp_valid$pred1)
  rmse2_tmp <- RMSE(tmp_valid$rating, tmp_valid$pred2)
  rmse3_tmp <- RMSE(tmp_valid$rating, tmp_valid$pred3)
  rmse4_tmp <- RMSE(tmp_valid$rating, tmp_valid$pred4)
  rmse5_tmp <- RMSE(tmp_valid$rating, tmp_valid$pred5)
  
  rmse_tmp_df <- rbind(data_frame(Model = 1, RMSE = rmse1_tmp),
                       data_frame(Model = 2, RMSE = rmse2_tmp),
                       data_frame(Model = 3, RMSE = rmse3_tmp),
                       data_frame(Model = 4, RMSE = rmse4_tmp),
                       data_frame(Model = 5, RMSE = rmse5_tmp)
  ) %>% mutate(ith_fold = i)
  
  #Create a table to stock the RMSE from each of the k cross validation steps
  #if first cycle, create the df. For all other cycles bind to the existing table.
  if(i==1){
    rmse_tmp_df_all <- rmse_tmp_df
  }
  else{
    rmse_tmp_df_all <- rbind(rmse_tmp_df_all, rmse_tmp_df)
  }
  
  #For the last cycle create a df and stock the mean RMSE from the k validation sets
  if(i==k){
    rmse_results <- rmse_tmp_df_all %>% group_by(Model) %>% summarize(RMSE = mean(RMSE))
  }
}

rmse_results %>% knitr::kable() #show rmse results
#So model 5 still an improvement, although less of a bump than in training set. Probably a degree of over-fitting.


######################################
# Testing best Model using holdout set
######################################

#Repeat data processing for edx on final holdout set
final_holdout_test_pr <- final_holdout_test %>% mutate(filmYear = str_sub(title,-6,-1),
                         title = str_sub(title,end=-7)) %>% 
  mutate(filmYear = gsub('[()]','',filmYear),
         ratingdate = as_datetime(timestamp)) %>%
  mutate(filmDecade = paste0(floor(as.integer(filmYear)/10)*10,'s'),
         Genres_Cnt = 1 + str_count(genres,"\\|")) %>%
  select(-timestamp)

for(n in 1:m){
  newcol <- genrelist[n]
  final_holdout_test_pr[newcol] <- grepl(newcol, final_holdout_test_pr$genres, fixed = TRUE)
}

rm(final_holdout_test) #remove unprocessed edx set for disk mgmt


#Get final model params from full edx_pr set
MinVol = 5

mu <- mean(edx_pr$rating)  
movie_avgs <- edx_pr %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
user_avgs <- edx_pr %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_gnre_avgs <- user_avgs %>% select(userId)   #Get user IDs in train set

#Loop through genres and create a set with each genre's bias for each user
for(x in genrelist){
  user_gnre_avgs_tmp <- edx_pr %>% select(movieId,userId,x,rating) %>%
    left_join(movie_avgs, by='movieId') %>%                             #to get b_i
    left_join(user_avgs, by='userId') %>%                               #to get b_u
    group_by_at(c('userId', x)) %>%                                     #group by user and genre
    summarize(n=n(), b_u_g = mean(rating - mu - b_i - b_u)) %>%         #get bias for the genre
    filter(get(x) == TRUE) %>%                                          #only retain figures for ratings related to the genre x
    filter(n >= MinVol) %>% select(userId, b_u_g)                       #only retain where user gave 5 ratings for genre x 
  
  names(user_gnre_avgs_tmp)[names(user_gnre_avgs_tmp) == "b_u_g"] <- paste0(x,"_b_u_g") #change b_u_g var name to "Action_b_u_g" etc
  
  user_gnre_avgs <- user_gnre_avgs %>% left_join(user_gnre_avgs_tmp, by='userId') #Join onto the df for all genres
  
}

#replace all NAs with zeroes (fewer than minvol ratings for the genre, so give them zero bias)
user_gnre_avgs <- user_gnre_avgs %>% replace(is.na(.), 0)

#Calc the overall average bias given the genres of each movie for a given user in the validation set
user_gnre_avgs_fin <- final_holdout_test_pr %>% select(c('userId', 'movieId', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                                             'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 
                                             'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')) %>% mutate() %>% 
  left_join(user_gnre_avgs,by='userId') %>% rename(FilmNoir = "Film-Noir", SciFi= "Sci-Fi",
                                                   FilmNoir_b_u_g = "Film-Noir_b_u_g", SciFi_b_u_g= "Sci-Fi_b_u_g") %>%
  mutate(b_u_g_num = Action*Action_b_u_g + Adventure*Adventure_b_u_g + Animation*Animation_b_u_g +
           Children*Children_b_u_g + Comedy*Comedy_b_u_g + Crime*Crime_b_u_g + Documentary*Documentary_b_u_g + 
           Drama*Drama_b_u_g + Fantasy*Fantasy_b_u_g +
           FilmNoir*FilmNoir_b_u_g + Horror*Horror_b_u_g + IMAX*IMAX_b_u_g +
           Musical*Musical_b_u_g + Mystery*Mystery_b_u_g + Romance*Romance_b_u_g +
           SciFi*SciFi_b_u_g + Thriller*Thriller_b_u_g + War*War_b_u_g +Western*Western_b_u_g ,
         b_u_g_denom = rowSums(across(c(Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, 
                                        FilmNoir, Horror, IMAX, Musical, Mystery, Romance, SciFi, Thriller, 
                                        War, Western)))) %>% 
  mutate(b_u_g = b_u_g_num / b_u_g_denom) %>% select(userId, movieId,b_u_g)


#Join the genre biases onto the test dataset and reset handful of NAs for one movie without genre to zero bias
final_holdout_test_pr <- final_holdout_test_pr %>% 
  left_join(movie_avgs,by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(user_gnre_avgs_fin, by=c('userId','movieId')) %>% 
  mutate(b_u_g = ifelse(is.na(b_u_g), 0, b_u_g)) %>% mutate(pred5 = mu + b_i + b_u + b_u_g)

#Calculate final RMSE in test set
RMSE(final_holdout_test_pr$pred5,final_holdout_test_pr$rating)


