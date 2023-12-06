 library(tidymodels)
 library(embed)
 library(vroom)
 library(tidyverse)
 library(themis)
 library(kknn)
 library(naivebayes)
 library(discrim)
 library(lme4)
 library(kernlab)

## Read in the data
amazonTrain <- vroom("amazon.train.csv")
amazonTest <- vroom("amazon.test.csv")

amazon_recipe <- recipe(ACTION~., data=amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_mutate(ACTION=as.factor(ACTION), skip=TRUE) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.5) %>%
  step_smote(all_outcomes(), neighbors=5)

# apply the recipe to your data
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazonTrain)

## Logistic Regression

my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazonTrain) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=amazonTest,
                              type="class") # "class" or "prob" (see doc)

head(amazon_predictions)

amazon_predictions <- cbind(amazonTest$id, amazon_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_class") #%>%

vroom_write(x=amazon_predictions, file="logisticpredstestID.csv", delim=",")

## Penalized Logistic Regression

# Define the model
my_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

# Create the workflow
amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(my_mod)

# Define the hyperparameter grid
tuning_grid <- grid_regular(penalty(), mixture(), levels = 3) # Adjust the number of levels as needed

# Split data for cross-validation
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Run cross-validation
CV_results <- amazon_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_wf %>%
  predict(new_data = amazonTrain, type="class")

preg_predictions <- predict(final_wf,
                            new_data=amazonTest,
                            type="class") # "class" or "prob" (see doc)

preg_predictions <- cbind(amazonTest$id, preg_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_class") #%>%

vroom_write(x=preg_predictions, file="pregID.csv", delim=",")

## Random Forest
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(my_mod)

# Set up grid of tuning values 
tuning_grid <- grid_regular(mtry(range =c(1,7)), min_n(), levels = 3) 

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_forest_wf <-
  forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_forest_wf %>%
  predict(new_data = amazonTrain, type="prob")

forest_predictions <- predict(final_forest_wf,
                              new_data=amazonTest,
                              type="prob") # "class" or "prob" (see doc)

forest_predictions <- cbind(amazonTest$id, forest_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%

vroom_write(x=forest_predictions, file="forestID.csv", delim=",")

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
# Set up grid of tuning values 
tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 3)

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_bayes_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_bayes_wf %>%
  predict(new_data = amazonTrain, type="prob")

bayes_predictions <- predict(final_bayes_wf,
                             new_data=amazonTest,
                             type="prob") # "class" or "prob" (see doc)

bayes_predictions <- cbind(amazonTest$id, bayes_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%

vroom_write(x=bayes_predictions, file="bayesID.csv", delim=",")

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(knn_model)

## Fit or Tune Model HERE
# Set up grid of tuning values 
tuning_grid <- grid_regular(neighbors(), levels = 3)

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_knn_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_knn_wf %>%
  predict(new_data = amazonTrain, type="prob")

knn_predictions <- predict(final_knn_wf,
                           new_data=amazonTest,
                           type="prob") # "class" or "prob" (see doc)

knn_predictions <- cbind(amazonTest$id, knn_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%

vroom_write(x=knn_predictions, file="knnID.csv", delim=",")

## SVM model
svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

# Create a workflow with model & recipe
svm_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svmLinear)

# Set up grid of tuning values 
tuning_grid <- grid_regular(cost(), levels = 3) 

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- svm_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_svm_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_svm_wf %>%
  predict(new_data = amazonTrain, type="prob")

svm_predictions <- predict(final_svm_wf,
                           new_data=amazonTest,
                           type="prob") # "class" or "prob" (see doc)

svm_predictions <- cbind(amazonTest$id, svm_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%

vroom_write(x=svm_predictions, file="svmID.csv", delim=",")

