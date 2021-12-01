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
library(tidyr)

# Reading in data
train = read.csv("C:/Users/patri/OneDrive/School Files/3rd Year College/STAT 338/Final Project/australia_train.csv")
test = read.csv("C:/Users/patri/OneDrive/School Files/3rd Year College/STAT 338/Final Project/australia_test.csv")

data.frame("NA Count" = sapply(train, function(x) {return(sum(is.na(x)))}))
str(train)

# Function to make it easier to create formulas for models when using lots of predictors
generate_formula = function(y, cols) {
  out = paste(y, "~", cols[1])
  for (col in cols[2:(length(cols))]) {
    out = paste(out, "+", col)
  }
  return(as.formula(out))
}

# Single function ensure the same preprocessing is done to both train and test sets
process = function(df) {
  # Fix data types
  # if ('rain_tomorrow' %in% names(df)) {
  #   df$rain_tomorrow = as.factor(df$rain_tomorrow)
  # }
  df$date = as.Date(df$date)
  
  # Filling in NA values (currently this just fills in NAs with data from previous day
  # but could replace this with a better method later)
  df = df %>% fill(min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir,
                   wind_gust_speed, wind_dir9am, wind_dir3pm, wind_speed9am, wind_speed3pm,
                   humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm,
                   temp9am, temp3pm, rain_today, .direction='downup')
  
  # Feature generation
  df[,'Delta_Wind_Speed'] = df$wind_speed3pm - df$wind_speed9am
  df[,'Delta_Humidity'] = df$humidity3pm - df$humidity9am
  df[,'Delta_Pressure'] = df$pressure3pm - df$pressure9am
  df[,'Delta_Cloud'] = df$cloud3pm - df$cloud9am
  df[,'Delta_Temp'] = df$temp3pm - df$temp9am
  
  # Dummy variables for categorical features (include location/region?)
  dummy_gen = dummyVars("~wind_gust_dir + wind_dir9am + wind_dir3pm", data = df)
  dummies = data.frame(predict(dummy_gen, df))
  df = cbind(df, dummies)
  
  # Drop original columns that were used for dummy vars
  to_remove = c("wind_dir3pm", "wind_dir9am", "wind_gust_dir", "location")
  df = df[,!names(df) %in% to_remove]
  
  # Select columns to normalize
  to_normalize = c("min_temp", "max_temp", "rainfall", "wind_gust_speed",
                   "wind_speed9am", "wind_speed3pm", "humidity9am", "humidity3pm",
                   "pressure9am", "pressure3pm", "cloud9am", "cloud3pm",
                   "temp9am", "temp3pm", "Delta_Wind_Speed", "Delta_Humidity",
                   "Delta_Pressure", "Delta_Cloud", "Delta_Temp")
  
  # Normalize data between 0 and 1
  df[,to_normalize] = sapply(df[,to_normalize], 
        function(x) {
          return((x-min(x))/(max(x)-min(x)))
        })
  
  # Binning continuous variables
  num_to_cat = data.frame(sapply(df[,to_normalize], cut, breaks=5, axis=1))
  dummy_gen_bin = dummyVars(generate_formula("", to_normalize), data = num_to_cat)
  dummies_bin = data.frame(predict(dummy_gen_bin, num_to_cat))
  df_binned = cbind(df[,!names(df) %in% to_normalize], dummies_bin)
  
  return(list(df, df_binned))
}

# Apply process method to train and test data
res_train = process(train)
train = res_train[[1]]
train_binned = res_train[[2]]
res_test = process(test)
test = res_test[[1]]
test_binned = res_test[[2]]

# Variable selection
full_formula = generate_formula("rain_tomorrow", 
                                names(train)[!names(train) %in% 
                                               c("id", "date", "rain_tomorrow")])
binned_formula = generate_formula("rain_tomorrow", 
                                  names(train_binned)[!names(train_binned) %in% 
                                                 c("id", "date", "rain_tomorrow")])
# LASSO
x_train = as.matrix(train[,names(train)[!names(train) %in% c("id", "date", "rain_tomorrow")]])
y = train$rain_tomorrow
lasso_select = cv.glmnet(x_train, y, alpha=1)
plot(lasso_select)
lasso_model = glmnet(x_train, y, alpha=1, lambda=lasso_select$lambda.1se)
lasso_model$beta

# CART
train$rain_tomorrow = as.factor(train$rain_tomorrow)
tree = rpart(full_formula, data = train)
rpart.plot(tree)
par(mar=c(4,13,1,1)) # Sets margins for plot
barplot(tree$variable.importance, horiz=T, las=2, main='Variable Importance', axes=F)

# Testing models
sample_train = train[sample(nrow(train), 5000),]
sample_train_binned = train_binned[sample(nrow(train_binned), 5000),]

# Random Forest
sample_train_binned$rain_tomorrow = as.factor(sample_train_binned$rain_tomorrow)
rf_model = randomForest(binned_formula, data = sample_train_binned)
rf_preds = predict(rf_model, sample_train_binned)
rf_accuracy = accuracy(sample_train_binned$rain_tomorrow, rf_preds)
plot(rf_model)
varImpPlot(rf_model)

# SVM
svm_model = svm(full_formula, data = sample_train,
                type='C-classification', cross=10)
svm_accuracy = accuracy(sample_train$rain_tomorrow, svm_model$fitted)


# k-fold cross validation

#### Random forest
df = sample_train_binned
k = 5
folds = sample(1:k, nrow(df), replace=T)
accuracy_list = rep(0, k)
logloss_list = rep(0, k)
for (i in 1:k) {
  train2 = df[folds!=i,]
  test2 = df[folds==i,]
  ct = randomForest(binned_formula, data = train2)
  preds = predict(ct, test2[,-c(which(names(df) == "rain_tomorrow"))])
  probs = predict(ct, test2[,-c(which(names(df) == "rain_tomorrow"))], type='prob')
  logloss = logLoss(test2$rain_tomorrow, probs[,2])
  logloss_list[i] = logloss
  accuracy = accuracy(test2$rain_tomorrow, preds)
  accuracy_list[i] = accuracy
}
cv_acc = mean(accuracy_list)
print(paste("Cross validation accuracy:", cv_acc))
cv_ll = mean(logloss_list)
print(paste("Cross validation log loss:", cv_ll))



#### Catboost (try using numeric vs binned features)
df_cat = train_binned
k = 10
folds = sample(1:k, nrow(df), replace=T)
logloss_list_cat = rep(0, k)
iters = seq(100, 300, by=5)
iters_list = data.frame(iters = iters, logloss = 0)
for (iter in iters) {
  for (i in 1:k) {
    train2 = df_cat[folds!=i,]
    test2 = df_cat[folds==i,]
    features = train2[,!names(train2) %in% c('id', 'date', 'rain_tomorrow')]
    labels = train2$rain_tomorrow
    train_pool = catboost.load_pool(data = as.matrix(features), label = labels)
    real_pool = catboost.load_pool(test2[,!names(test2) %in% c('id', 'date')])
    model <- catboost.train(train_pool,  NULL,
                            params = list(loss_function = 'Logloss',
                                          iterations = iter,
                                          prediction_type = 'Probability',
                                          class_weights=c(.5,2), logging_level='Silent'))
    prediction = catboost.predict(model, real_pool, prediction_type='Probability')
    logloss = logLoss(as.numeric(test2$rain_tomorrow), prediction)
    logloss_list_cat[i] = logloss
  }
  cv_ll_cat = mean(logloss_list_cat)
  print(paste("Cross validation log loss:", cv_ll_cat, "#Iterations:", iter))
  iters_list[iters_list$iters == iter,'logloss'] = cv_ll_cat
}
ggplot(data = iters_list, aes(x=iters, y=logloss)) + geom_point() + geom_smooth()

