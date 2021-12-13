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
library(mice)

# Reading in data
train = read.csv("C:/Users/patri/Desktop/R Studio Files/STAT-338-Final/australia_train.csv")
test = read.csv("C:/Users/patri/Desktop/R Studio Files/STAT-338-Final/australia_test.csv")

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
  df$evaporation = as.integer(df$evaporation)
  df$sunshine = as.integer(df$sunshine)
  
  # Translate city to state for location
  in_victoria = c('Sale', 'Nhil', 'Watsonia', 'Richmond', 'Portland', 'Dartmoor',
                  'Ballarat', 'Bendigo', 'MelbourneAirport', 'Mildura', 'Melbourne')
  in_nsw = c('BadgerysCreek', 'WaggaWagga', 'Wollongong', 'Moree', 'SydneyAirport',
             'Sydney', 'Albury', 'Penrith', 'CoffsHarbour', 'Williamtown', 'Cobar',
             'NorahHead', 'Newcastle')
  in_queensland = c('Townsville', 'GoldCoast', 'Brisbane', 'Cairns')
  in_northern_territory = c('Uluru', 'AliceSprings', 'Katherine', 'Darwin')
  in_south_australia = c('Nuriootpa', 'MountGambier', 'Woomera', 'Adelaide')
  in_west_australia = c('Albany', 'Perth', 'PearceRAAF', 'Walpole', 'SalmonGums', 
                        'PerthAirport', 'Witchcliffe')
  in_capital = c('Tuggeranong', 'MountGinini', 'Canberra')
  in_other = c('Launceston', 'Hobart', 'NorfolkIsland')
  
  df[df$location %in% in_victoria,'state'] = 'Victoria'
  df[df$location %in% in_nsw,'state'] = 'New South Wales'
  df[df$location %in% in_queensland,'state'] = 'Queensland'
  df[df$location %in% in_northern_territory,'state'] = 'Northern Territory'
  df[df$location %in% in_south_australia,'state'] = 'South Australia'
  df[df$location %in% in_west_australia,'state'] = 'West Australia'
  df[df$location %in% in_capital,'state'] = 'Capital Territory'
  df[df$location %in% in_other,'state'] = 'Other'
  
  # Filling in NA values (uses mice first and then LOCF for anything that mice misses)
  mids = mice(df)
  df = complete(mids)
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
  dummy_gen = dummyVars("~wind_gust_dir + wind_dir9am + wind_dir3pm + state", data = df)
  dummies = data.frame(predict(dummy_gen, df))
  df = cbind(df, dummies)
  
  # Drop original columns that were used for dummy vars
  to_remove = c("wind_dir3pm", "wind_dir9am", "wind_gust_dir", "location", 'state')
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

# Plots
mosaicplot(rain_tomorrow~rain_today, data=train)
mosaicplot(rain_tomorrow~sunshine, data=train)
mosaicplot(rain_tomorrow~evaporation, data=train)

ggplot(data = train, aes(x=humidity3pm, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=rain_today, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=Delta_Humidity, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=wind_gust_speed, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=cloud9am, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=cloud3pm, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=Delta_Temp, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=Delta_Cloud, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=pressure3pm, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=wind_speed3pm, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=Delta_Pressure, y=rain_tomorrow)) + geom_violin()
ggplot(data = train, aes(x=rainfall, y=rain_tomorrow)) + geom_point()

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
# train$rain_tomorrow = as.factor(train$rain_tomorrow)
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

rf_predictors = c("humidity3pm", "rain_today", "Delta_Humidity", "wind_gust_speed",
                  "cloud9am", "cloud3pm", "Delta_Temp", "Delta_Cloud", "pressure3pm",
                  "wind_speed3pm", "Delta_Pressure", "stateNew.South.Wales", 
                  "Delta_Wind_Speed", "sunshine")

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



#### Catboost tuning
df_cat = train
k = 10
folds = sample(1:k, nrow(df_cat), replace=T)
depths = seq(3.5, 5, by=0.1)
depths_list = data.frame(depth = depths, logloss = 0)
for (depth in depths) {
  features = df_cat[,!names(df_cat) %in% c('id', 'date', 'rain_tomorrow')]
  labels = as.integer(df_cat$rain_tomorrow)
  train_pool = catboost.load_pool(data = as.matrix(features), label = labels)
  res = catboost.cv(train_pool,  fold_count=k, stratified=TRUE,
                    params = list(loss_function = 'Logloss',
                                  iterations = 600,
                                  prediction_type = 'Probability',
                                  depth=8,
                                  class_weights=c(depth,1), logging_level='Silent'))
  cv_ll_cat = min(res$test.Logloss.mean)
  print(paste("Cross validation log loss:", cv_ll_cat))
  depths_list[depths_list$depth == depth,'logloss'] = cv_ll_cat
}
ggplot(data = depths_list, aes(x=depth, y=logloss)) + geom_point() + geom_smooth()

# Testing different feature sets
cv_cat = function(x_train, y_train, iters=600, k=5, depth=8, weight=3.5) {
  features = x_train[,!names(x_train) %in% c('id', 'date', 'rain_tomorrow')]
  labels = as.integer(y_train)
  train_pool = catboost.load_pool(data = as.matrix(features), label = labels)
  results = catboost.cv(train_pool,  fold_count=k, stratified=FALSE,
                        params = list(loss_function = 'Logloss',
                                      iterations = iters,
                                      prediction_type = 'Probability',
                                      depth=depth,
                                      class_weights=c(weight,1), logging_level='Silent'))
  return(results)
}
df1 = train # All predictors
df2 = train[,colnames(x_train)[which(!lasso_model$beta == 0)]] # Lasso selected predictors
df3 = train_binned[,rf_model$importance[order(rf_model$importance),1] %>% tail(50) %>% names] # Random forest predictors (20 highest importance from rf_model)
df4 = train[,rf_predictors] # Unbinned predictors from random forest
df5 = cbind(df3, df4) # Combination of both binned and unbinned predictors
dfs = list('FULL' = df1, 'LASSO' = df2, 'BINNED RF' = df3, 'UNBINNED RF' = df4, 'COMBO' = df5)
y = train$rain_tomorrow
counter = 1
for (df in dfs) {
  res = cv_cat(df, y, iters=1000)
  min_ll = min(res$test.Logloss.mean)
  print(ggplot(data = res, aes(x=c(1:nrow(res)), y=test.Logloss.mean)) + geom_line() +
          geom_ribbon(aes(ymin=test.Logloss.mean-test.Logloss.std, 
                          ymax=test.Logloss.mean+test.Logloss.std,
                          fill='red'), alpha=0.3) + 
          labs(title=paste('Cross validation error over training iterations using', 
                           names(dfs[counter]), 'dataset'),
               x='Iteration', y='Test Logloss Mean', fill='Std. Err.', 
               caption=paste('Min logLoss:',min_ll)))
  print(paste('Min logloss for', names(dfs[counter]), 'dataset:', min_ll))
  print(paste(names(dfs[counter]), 'reached min after:', 
              which(res$test.Logloss.mean == min_ll), 'iterations'))
  counter = counter + 1
}


#### Logistic regression
logistic_model = glm(y~x_train, family='binomial')

# Submission generation code 
features = df1[,!names(df1) %in% c('id', 'date', 'rain_tomorrow')]
labels = as.integer(y)
train_pool = catboost.load_pool(data = as.matrix(features), label = labels)
real_pool = catboost.load_pool(data = as.matrix(test[,!names(test) %in% c('id', 'date')]))
final_model = catboost.train(train_pool,
                          params = list(loss_function = 'Logloss',
                                        iterations = 900,
                                        prediction_type = 'Probability',
                                        depth=8,
                                        class_weights=c(1.2,1), logging_level='Silent'))
predicted_final = catboost.predict(final_model, real_pool, prediction_type='Probability')
submission = data.frame(id=test$id, rain_tomorrow = predicted_final)
write.csv(submission, 'australia_submission.csv', row.names=F)
