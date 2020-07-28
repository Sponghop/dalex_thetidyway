library(tidymodels)
library(mlbench)
library(caret)
library(xgboost)
library(doFuture)
library(modelStudio)

source("scripts/custom_functions.R")

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

## Dalex package
explainer_xgboost <- DALEX::explain(
  model = custom_model_expl(reg_res),
  data = custom_data_expl(reg_res, pid_train, "diabetes"),
  y = custom_y_expl(reg_res, pid_train, "diabetes"),
  predict_function = custom_predict,
  label = "xgboost")

## Residual diagnostics
resids_xgboost <- DALEX::model_performance(explainer_xgboost)
p1 <- plot(resids_xgboost)
p2 <- plot(resids_xgboost, geom = "boxplot")
gridExtra::grid.arrange(p1, p2, nrow = 1)

## Variable Importance
vip_xgboost <- DALEX::variable_importance(explainer_xgboost, loss_function = DALEX::loss_root_mean_square) 
plot(vip_xgboost, max_vars = 10)

## Break-down Plots for Additive Attributions
new_observation <- custom_new_obs(reg_res, pid_train, "diabetes", 1)

## Break-down Plots for Interactions
bd_xgboost <- breakDown::break_down(explainer_xgboost, new_observation, keep_distributions = TRUE)
plot(bd_xgboost)

## Shapley values
shap_new <- DALEX::variable_attribution(explainer_xgboost, new_observation, type = "shap")
plot(shap_new) 

## Local model
local_model_xgboost <- localModel::individual_surrogate_model(explainer_xgboost, new_observation, size = 1000, seed = 1313)
plot(local_model_xgboost)

## Ceteris Paribus Explainer
cp_pp_xgboost <- DALEX::predict_profile(explainer_xgboost, new_observation)
plot(cp_pp_xgboost, variables = c("age", "glucose"))

## Dashboard overzicht
modelStudio::modelStudio(explainer_xgboost)

