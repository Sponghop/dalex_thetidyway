###Model Training/Tuning XGBoost
randomforest_mod <-
  rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

ml_wflow <-
  workflow() %>%
  add_recipe(mod_rec) %>%
  add_model(randomforest_mod)

ctrl <- control_resamples(save_pred = TRUE)
folds <- vfold_cv(pid_train, v = 2, repeats = 3)
grid <-  expand.grid(mtry = 1:6, trees = c(250,500,750))

all_cores <- parallel::detectCores(logical = TRUE) - 1
registerDoFuture()
cl <- makeCluster(all_cores)
plan(future::cluster, workers = cl)

res <- 
  ml_wflow %>%
  tune_grid(resamples = folds, control = ctrl, grid = grid)

stopCluster(cl)

res %>%
  tune::collect_metrics()

best_params <-
  res %>%
  tune::select_best(metric = "accuracy")
best_params

## Validation
reg_res_randomforest <-
  ml_wflow %>%
  # Attach the best tuning parameters to the model
  finalize_workflow(best_params) %>%
  # Fit the final model to the training data
  fit(data = pid_train)

pid_test <- testing(pid_split)

reg_res_randomforest %>%
  predict(new_data = pid_test) %>%
  bind_cols(pid_test, .) %>%
  select(diabetes, .pred_class) %>% 
  accuracy(diabetes, .pred_class)