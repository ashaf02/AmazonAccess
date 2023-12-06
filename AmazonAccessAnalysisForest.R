library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(themis)

## Read in the data
amazonTrain <- vroom("amazon.train.csv")
amazonTest <- vroom("amazon.test.csv")

amazon_recipe <- recipe(ACTION~., data=amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_mutate(ACTION=as.factor(ACTION), skip=TRUE)# %>%
  # step_normalize(all_predictors()) %>%
  # step_pca(all_predictors(), threshold=.85) %>%
  # step_smote(all_outcomes(), neighbors=5)

head(amazon_recipe)

# apply the recipe to your data
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazonTrain)

## Random Forest
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=600) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(my_mod)

# Set up grid of tuning values 
tuning_grid <- grid_regular(mtry(range =c(1,7)), min_n(), levels = 4) 

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, precision),
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

## Predict
final_forest_wf %>%
  predict(new_data = amazonTrain, type="prob")

forest_predictions <- predict(final_forest_wf,
                              new_data=amazonTest,
                              type="prob") # "class" or "prob" (see doc)

forest_predictions <- cbind(amazonTest$id, forest_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1")

vroom_write(x=forest_predictions, file="realforestfinal.csv", delim=",")
