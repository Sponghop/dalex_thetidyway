library(tidymodels)
library(mlbench)
library(rpart)
library(caret)
library(rpart.plot)
library(xgboost)
library(doParallel)

## Data inladen & train/test split
data(PimaIndiansDiabetes)
PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
set.seed(20200605)
pid <- rsample::initial_split(PimaIndiansDiabetes)
pid_train <- rsample::training(pid)
pid_test <- rsample::testing(pid)

## Boosting Machine
PimaIndian_rec <-
  recipe(diabetes ~ ., data = pid_train) %>%
  step_normalize(all_predictors())

PimaIndian_rec %>% 
  parameters() %>% 
  pull("object")

xgboost_mod <-
  boost_tree(mtry = tune(), tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification") #%>%
  #fit(diabetes ~., data = pid_train)

ctrl <- control_resamples(save_pred = TRUE)
folds <- vfold_cv(pid_train, v = 2, repeats = 3)
grid <-  expand.grid(mtry = 3:6, tree_depth = 4:5)

xgboost_res <- tune_grid(xgboost_mod, PimaIndian_rec, resamples = folds, control = ctrl, grid = grid) 
show_best(xgboost_res, metric = "accuracy")
select_best(xgboost_res, metric = "accuracy")

xg_wflow <-
  workflow() %>%
  add_model(xgboost_mod) %>%
  add_recipe(PimaIndian_rec) 
xg_wflow

xg_param_final <- select_by_one_std_err(xgboost_res, mtry, tree_depth, metric = "accuracy")
xg_wflow_final <- finalize_workflow(xg_wflow, xg_param_final)
xg_wflow_final_fit <- fit(xg_wflow_final, data = pid_train)
xg_pim_rec <- pull_workflow_prepped_recipe(xg_wflow_final_fit)

xg_final_fit <- pull_workflow_fit(xg_wflow_final_fit)
pid_test$pred <- predict(xg_final_fit, new_data = bake(xg_pim_rec, pid_test))$.pred


confusionMatrix(data = klasse_voorspelling_test$.pred_class,
                reference = pid_test$diabetes,
                positive = "pos")
