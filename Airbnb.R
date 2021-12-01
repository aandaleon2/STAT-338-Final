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


### TODOs
# 1) Standardize data
# 2) Try out different feature sets + hyperparameters for models
# 3) Apply the transformations from EDA section to models
# 4) Add cross validation to model testing section
# 5) Generate new features from data to experiment with
#   5a) Try out NLP model(s) on 'name' feature


# Read in data
train = read.csv("C:/Users/patri/OneDrive/School Files/3rd Year College/STAT 338/Final Project/airbnb_train.csv")
test = read.csv("C:/Users/patri/OneDrive/School Files/3rd Year College/STAT 338/Final Project/airbnb_test.csv")

predictors_num = c('latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                   'reviews_per_month', 'calculated_host_listings_count', 'reviews_per_month',
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
# Fix data types
train$last_review = as.Date(train$last_review)
test$last_review = as.Date(test$last_review)

# Dealing with NA values
data.frame(sapply(train, function(x) {return(sum(is.na(x)))}))
data.frame(sapply(test, function(x) {return(sum(is.na(x)))}))
test[is.na(test$reviews_per_month),'reviews_per_month'] = 0
test[is.na(test$last_review), 'last_review'] = mean(test$last_review, na.rm=T)
train[is.na(train$reviews_per_month),'reviews_per_month'] = 0
train[is.na(train$last_review), 'last_review'] = mean(train$last_review, na.rm=T)

# Dummy variables
dummy_gen_train = dummyVars("~neighbourhood_group + room_type", data = train)
dummies_train = data.frame(predict(dummy_gen_train, train))
train = cbind(train, dummies_train)

dummy_gen_test = dummyVars("~neighbourhood_group + room_type", data = test)
dummies_test = data.frame(predict(dummy_gen_test, test))
test = cbind(test, dummies_test)

# Binning continuous variables 
# note: right now binning is done separately, need to make sure train/test have same bins
num_to_cat_train = data.frame(sapply(train[,predictors_num], cut, breaks=5, axis=1))
dummy_gen_bin_train = dummyVars(generate_formula("", predictors_num), data = num_to_cat_train)
dummies_bin_train = data.frame(predict(dummy_gen_bin_train, num_to_cat_train))
train = cbind(train, dummies_bin_train)

num_to_cat_test = data.frame(sapply(test[,predictors_num], cut, breaks=5, axis=1))
dummy_gen_bin_test = dummyVars(generate_formula("", predictors_num), data = num_to_cat_test)
dummies_bin_test = data.frame(predict(dummy_gen_bin_test, num_to_cat_test))
test = cbind(test, dummies_bin_test)

# Test area for NLP stuff
train[,'Cozy_name'] = sapply(train$name, str_contains, pattern='cozy', ignore.case=TRUE)
train[,'Private_name'] = sapply(train$name, str_contains, pattern='private', ignore.case=TRUE)
test[,'Cozy_name'] = sapply(test$name, str_contains, pattern='cozy', ignore.case=TRUE)
test[,'Private_name'] = sapply(test$name, str_contains, pattern='private', ignore.case=TRUE)


# EDA and feature selection -----------------------------------------------
predictors_binned = names(dummies_bin_train)
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
selected_predictors = c(predictors_num, predictors_cat)
new_formula = as.formula("logprice ~ latitude + log(abs(longitude)) + sqrt(minimum_nights) + 
                  reviews_per_month + sqrt(calculated_host_listings_count) + 
                  availability_365 + number_of_reviews + neighbourhood_groupBronx +
                  neighbourhood_groupManhattan + neighbourhood_groupBrooklyn + 
                  neighbourhood_groupQueens + neighbourhood_groupStaten.Island +
                  room_typeEntire.home.apt + room_typePrivate.room + Cozy_name + Private_name")

# SVM (change change type, nu, gamma, degree, kernel, and cost)
svm_model = svm(new_formula,
                data = train2, type='nu-regression', kernel='polynomial',
                degree=3, nu=.4, gamma=.1)
predicted_svm = predict(svm_model, test2)
print(paste("RMSE for SVM model:", rmse(actual, exp(predicted_svm))))

# XGBoost
xg_model = xgboost(data = as.matrix(train2[,selected_predictors]),
                   label = train2$price, nrounds = 50)
predicted_xg = predict(xg_model, as.matrix(test2[,selected_predictors]))
print(paste("RMSE for XG model:", rmse(actual, predicted_xg)))

# Random Forest (note: currently using all categorical variables for quicker training)
binned_predictors = c(predictors_cat, predictors_binned)
rf_model = randomForest(generate_formula('price', binned_predictors),
                        data = train2)
predicted_rf = predict(rf_model, test2[,binned_predictors])
print(paste("RMSE for random forest model:", rmse(actual, predicted_rf)))
varImpPlot(rf_model)

# GAM (using top features from varImpPlot of random forest and cart models)
gam_model = gam(logprice ~ s(latitude) + s(log(abs(longitude))) + room_typeEntire.home.apt +
                  neighbourhood_groupManhattan + availability_365..0.365.73. + 
                  Private_name + room_typePrivate.room + s(sqrt(minimum_nights)) + 
                  s(reviews_per_month), data = train2)
predicted_gam = predict(gam_model, test2)
print(paste("RMSE for GAM model:", rmse(actual, exp(predicted_gam))))

# Cross validation GAM
gam_model_cv = CVgam(logprice ~ s(latitude) + s(log(abs(longitude))) + room_typeEntire.home.apt +
                       neighbourhood_groupManhattan + availability_365..0.365.73. + 
                       Private_name + room_typePrivate.room + s(sqrt(minimum_nights)) + 
                       s(reviews_per_month), data=train2, nfold=10)
print(paste("CV RMSLE for GAM model:", rmsle(train2$price, exp(gam_model_cv$fitted))))

# Submission generation code (update with whatever model is used and potentially remove exp transformation)
#predicted_final = predict(rf_model, test)
#submission = data.frame(id=test$id, price = exp(predicted_final))
#write.csv(submission, 'airbnb_submission.csv', row.names=F)


