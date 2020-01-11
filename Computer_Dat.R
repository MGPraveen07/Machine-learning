# load data set
dataset<- read.csv('Computer_data.csv')

# Encoding categorical variabal
dataset$cd <- factor(dataset$cd,
                     levels = c('yes', 'no'),
                     labels= c(1,0))
dataset$multi <- factor(dataset$multi,
                     levels = c('yes', 'no'),
                     labels= c(1,0))
dataset$premium <- factor(dataset$premium,
                     levels = c('yes', 'no'),
                     labels= c(1,0))

# Splitting the data set
library(caTools)
set.seed(100)
split = sample.split(dataset$price, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Fitting liner model to linear regression to the training set
regressor <- lm(formula = price ~., 
                data = training_set)
summary(regressor)

# Predicting the test set results
y_pred <- predict(regressor, newdata =  test_set)

# Bulding optimal model for using backward elimination
regressor <- lm(formula = price ~ speed + hd +ram + screen + cd + multi +premium + ads +trend, 
                data = training_set)
summary(regressor)
