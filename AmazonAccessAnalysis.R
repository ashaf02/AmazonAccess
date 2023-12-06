# Hint: 112
library(tidymodels)
library(embed)
library(vroom)

## Read in the data
amazonTrain <- vroom("amazon.train.csv")
amazonTest <- vroom("amazon.test.csv")

amazon_recipe <- recipe(ACTION~., data=amazonTrain) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>%# dummy variable encoding
  step_mutate(ACTION=as.factor(ACTION), skip=TRUE)
  

# NOTE: some of these step functions are not appropriate to use together

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

amazon_predictions <- cbind(amazonTest$id, amazon_predictions) %>%
  rename(Id = "amazonTest$id",
  Action = ".pred_class") #%>%
  #select (-"amazonTest$id")

head(amazon_predictions)

vroom_write(x=amazon_predictions, file="logisticpreds3.csv", delim=",")
# save(file="filename.RData", list=c("logReg_wf"))
# load("filename.RData")

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

# View the results
CV_results

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best("roc_auc")
bestTune

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
#select (-"amazonTest$id")

vroom_write(x=preg_predictions, file="preg2.csv", delim=",")
