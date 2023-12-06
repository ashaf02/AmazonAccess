library(naivebayes)
library(discrim)
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(kknn)

## Read in the data
amazonTrain <- vroom("amazon.train.csv")
amazonTest <- vroom("amazon.test.csv")

amazon_recipe <- recipe(ACTION~., data=amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_mutate(ACTION=as.factor(ACTION), skip=TRUE) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9)

# apply the recipe to your data
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazonTrain)

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

bayes_predictions_pcdr <- predict(final_bayes_wf,
                             new_data=amazonTest,
                             type="prob") # "class" or "prob" (see doc)

bayes_predictions_pcdr <- cbind(amazonTest$id, bayes_predictions_pcdr) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%
#select (-"amazonTest$id")

vroom_write(x=bayes_predictions_pcdr, file="bayesPCDR.csv", delim=",")

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

knn_predictions_pcdr <- predict(final_knn_wf,
                           new_data=amazonTest,
                           type="prob") # "class" or "prob" (see doc)

knn_predictions_pcdr <- cbind(amazonTest$id, knn_predictions_pcdr) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%
#select (-"amazonTest$id")

vroom_write(x=knn_predictions_pcdr, file="knnPCDR.csv", delim=",")
