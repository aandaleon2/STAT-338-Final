library(caret)
library(ggplot2)
library(sjmisc)
library(dplyr)
library(rpart)
library(rpart.plot)
library(glmnet)
library(Metrics)
library(e1071)
library(xgboost)
library(catboost)
library(randomForest)
library(gam)
library(gamclass)

# Read in data
setwd("C:/Users/patri/Desktop/R Studio Files/STAT-338-Final")
train = read.csv("airbnb_train.csv")
test = read.csv("airbnb_test.csv")

predictors_num = c('latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                   'reviews_per_month', 'calculated_host_listings_count',
                   'availability_365')
predictors_cat = c('neighbourhood_groupBronx', 'neighbourhood_groupManhattan',
                   'neighbourhood_groupBrooklyn', 'neighbourhood_groupQueens',
                   'neighbourhood_groupStaten.Island', 'room_typeEntire.home.apt',
                   'room_typePrivate.room', 'room_typeShared.room', 
                   'Cozy_name', 'Private_name')

# Function to make it easier to create formulas for models when using lots of predictors
generate_formula = function(y, cols) {
  out = paste(y, "~", cols[1])
  for (col in cols[2:(length(cols))]) {
    out = paste(out, "+", col)
  }
  return(as.formula(out))
}

# Set seed for reproducibility
set.seed(123456789)


# Data prep ---------------------------------------------------------------
process = function(df) {
  # Fix data types
  df$last_review = as.Date(df$last_review)
  
  # Dealing with NA values
  df[is.na(df$reviews_per_month),'reviews_per_month'] = 0
  df[is.na(df$last_review), 'last_review'] = mean(df$last_review, na.rm=T)
  
  # Dummy variables
  dummy_gen = dummyVars("~neighbourhood_group + room_type", data = df)
  dummies = data.frame(predict(dummy_gen, df))
  df = cbind(df, dummies)
  
  # Binning continuous variables
  # num_to_cat = data.frame(sapply(df[,predictors_num], cut, breaks=5, axis=1))
  # dummy_gen_bin = dummyVars(generate_formula("", predictors_num), data = num_to_cat)
  # dummies_bin = data.frame(predict(dummy_gen_bin, num_to_cat))
  # df = cbind(df, dummies_bin)
  
  # Extract features from name
  df[,'Cozy_name'] = sapply(df$name, str_contains, pattern='cozy', ignore.case=TRUE)
  df[,'Private_name'] = sapply(df$name, str_contains, pattern='private', ignore.case=TRUE)
  
  # Normalize continuous vars
  normalized = data.frame(sapply(df[predictors_num], scale))
  df = cbind(normalized, df[,!(names(df) %in% c(predictors_num))])
  
  # Remove unused features
  to_remove = c("host_name", "neighbourhood_group", "neighbourhood", "room_type",
                "name", "last_review")
  df = df[,!(names(df) %in% to_remove)]
  
  return(df)
}
train = process(train)
test = process(test)

# EDA and feature selection -----------------------------------------------
cor(train[,predictors_num], train$price)
cor(train[,predictors_num], train$price^.5)

# Focusing on the highest importance variables from below
pairs(train[,c('longitude', 'minimum_nights', 'calculated_host_listings_count',
               'reviews_per_month')],train$price)

ggplot(data = train, aes(x = log(abs(longitude)), y = log(price))) + 
  geom_point() + geom_smooth()
ggplot(data = train, aes(x=sqrt(minimum_nights), y=log(price))) + geom_point() + geom_smooth()
ggplot(data = train, aes(x=reviews_per_month, y=log(price))) + geom_point() + geom_smooth() # Note: one major outlier x value
ggplot(data = train, aes(x=sqrt(calculated_host_listings_count), y=log(price))) + geom_point() + geom_smooth()

ggplot(data = train, aes(x=room_typePrivate.room, y=log(price))) + geom_violin()
ggplot(data = train, aes(x=room_typeEntire.home.apt, y=log(price))) + geom_violin()

# CART model for feature selection
tree = rpart(generate_formula('price', predictors_num), data = train)
rpart.plot(tree)

tree2 = rpart(generate_formula('price', predictors_cat), data = train)
rpart.plot(tree2)

tree3 = rpart(generate_formula('price', c(predictors_num,predictors_cat)), data = train)
par(mar=c(5,13,8,1)) # Sets margins for plot
rpart.plot(tree3)
par(mar=c(1,13,1,1)) # Sets margins for plot
barplot(tree3$variable.importance, horiz=T, las=2, main='Variable Importance', axes=F)

# LASSO model for feature selection
x_train_numeric = train[,predictors_num]
x_train_categorical = train[,predictors_cat]
x_train = train[,c(predictors_cat, predictors_num)] # Can change predictors here to use binned instead of continuous
y = train$price
lasso_select = cv.glmnet(as.matrix(x_train), y, alpha=1)
plot(lasso_select)
lasso_model = glmnet(as.matrix(x_train), y, alpha=1, lambda=lasso_select$lambda.min)
lasso_model$beta


# Test Models -------------------------------------------------------------
# Note: replace train2/test2 with cross validation later
cutoff = round(nrow(train) * .75)
train2 = train[1:cutoff,]
test2 = train[(cutoff+1):nrow(train),]
actual = test2$price
train2[,'logprice'] = log(train2$price)
train2[train2$logprice == -Inf,'logprice'] = 0.000001

# Change this to compare different feature sets
all_predictors = c(predictors_num, predictors_cat)
lasso_predictors = names(x_train)[as.vector(lasso_model$beta) != 0]
transformed_formula = as.formula("price ~ latitude + log(abs(longitude)) + sqrt(abs(minimum_nights)) + 
                  reviews_per_month + sqrt(abs(calculated_host_listings_count)) + 
                  availability_365 + number_of_reviews + neighbourhood_groupBronx +
                  neighbourhood_groupManhattan + neighbourhood_groupBrooklyn + 
                  neighbourhood_groupQueens + neighbourhood_groupStaten.Island +
                  room_typeEntire.home.apt + room_typePrivate.room + Cozy_name + Private_name")
logtrain = train
logtrain$price = log(logtrain$price)
logtrain[logtrain$price == -Inf,'price'] = 0.000001


# SVM (change change type, nu, gamma, degree, kernel, and cost)
svm_model = svm(generate_formula('price', all_predictors), data = train, cross=5)
predicted_svm = predict(svm_model, train)
print(paste("RMSE for SVM model:", rmsle(train$price, predicted_svm)))

# Random Forest 
rf_model = randomForest(generate_formula('price', all_predictors), data = train)
predicted_rf = predict(rf_model, train)
print(paste("RMSE for random forest model:", rmse(train$price, predicted_rf)))
varImpPlot(rf_model)

# GAM (using top features from varImpPlot of random forest and cart models)
# gam_model = gam(price ~ s(latitude) + s(log(abs(longitude))) + room_typeEntire.home.apt +
#                   neighbourhood_groupManhattan + availability_365..0.365.73. + 
#                   Private_name + room_typePrivate.room + s(sqrt(minimum_nights)) + 
#                   s(reviews_per_month), data = train)
gam_model = gam(transformed_formula, data = logtrain)
predicted_gam = predict(gam_model, train)
print(paste("RMSE for GAM model:", rmse(train$price, exp(predicted_gam))))

# Tuning random forest
results = data.frame("nodesize" = 0, "oob" = 0)
nodesizes = 3:10
counter = 1
for (n in nodesizes) {
  rf_tune = randomForest(generate_formula('price', all_predictors), data = train,
                         nodesize = n, mtry = m)
  results[counter,] = c(m, n, mean(rf_tune$oob.times))
  print(paste("M:", m, "N:", n, "OOB:", mean(rf_tune$oob.times)))
  counter = counter + 1
}

# Final model
rf_final = randomForest(generate_formula('price', all_predictors), data = train,
                        nodesize = 10)
plot(rf_final)
varImpPlot(rf_final)

# Submission generation code (update with whatever model is used and potentially remove exp transformation)
predicted_final = predict(rf_final, test)
submission = data.frame(id=test$id, price = predicted_final)
write.csv(submission, 'airbnb_submission.csv', row.names=F)


