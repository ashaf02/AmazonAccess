#Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
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
  step_pca(all_predictors(), threshold=.5)

# apply the recipe to your data
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazonTrain)

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
#select (-"amazonTest$id")

vroom_write(x=svm_predictions, file="svm.csv", delim=",")
