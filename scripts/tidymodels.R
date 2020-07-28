library(tidymodels)
library(mlbench)
library(caret)
library(xgboost)
library(doFuture)
library(modelStudio)

## Data preparatie
data(PimaIndiansDiabetes)
set.seed(1234)
pid_split <- initial_split(PimaIndiansDiabetes, prop = .75)
pid_train <- training(pid_split)

## Preprocessing
mod_rec <-
  recipe(diabetes ~ ., data = pid_train) %>%
  step_mutate(was_pregnant = as.factor(ifelse(pregnant > 0, 'Ja', 'Nee'))) %>%
  step_dummy(was_pregnant) %>%
  step_normalize(all_numeric())

## Model Training/Tuning
xgboost_mod <-
  boost_tree(mtry = tune(), tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

ml_wflow <-
  workflow() %>%
  add_recipe(mod_rec) %>%
  add_model(xgboost_mod)

ctrl <- control_resamples(save_pred = TRUE)
folds <- vfold_cv(pid_train, v = 2, repeats = 3)
grid <-  expand.grid(mtry = 3:11, tree_depth = 3:11)

all_cores <- parallel::detectCores(logical = TRUE) - 1
registerDoFuture()
cl <- makeCluster(all_cores)
plan(future::cluster, workers = cl)

res <- 
  ml_wflow %>%
  tune_grid(resamples = folds, control = ctrl, grid = grid)

res %>%
  tune::collect_metrics()

best_params <-
  res %>%
  tune::select_best(metric = "accuracy")
best_params

## Validation
reg_res <-
  ml_wflow %>%
  # Attach the best tuning parameters to the model
  finalize_workflow(best_params) %>%
  # Fit the final model to the training data
  fit(data = pid_train)

pid_test <- testing(pid_split)

reg_res %>%
  predict(new_data = pid_test) %>%
  bind_cols(pid_test, .) %>%
  select(diabetes, .pred_class) %>% 
  accuracy(diabetes, .pred_class)
# https://cimentadaj.github.io/blog/2020-02-06-the-simplest-tidy-machine-learning-workflow/the-simplest-tidy-machine-learning-workflow/

