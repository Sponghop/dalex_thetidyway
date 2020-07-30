## Dalex package
explainer_xgboost <- DALEX::explain(
  model = custom_model_expl(reg_res_xgboost),
  data = custom_data_expl(reg_res_xgboost, pid_train, "diabetes"),
  y = custom_y_expl(reg_res_xgboost, pid_train, "diabetes"),
  predict_function = custom_predict,
  label = "xgboost")

explainer_decisiontree <- DALEX::explain(
  model = custom_model_expl(reg_res_decisiontree),
  data = custom_data_expl(reg_res_decisiontree, pid_train, "diabetes"),
  y = custom_y_expl(reg_res_decisiontree, pid_train, "diabetes"),
  predict_function = custom_predict,
  label = "decisiontree")

explainer_randomforest <- DALEX::explain(
  model = custom_model_expl(reg_res_randomforest),
  data = custom_data_expl(reg_res_randomforest, pid_train, "diabetes"),
  y = custom_y_expl(reg_res_randomforest, pid_train, "diabetes"),
  predict_function = custom_predict,
  label = "randomforest")

## Residual diagnostics
resids_decisiontree <- DALEX::model_performance(explainer_decisiontree)
resids_randomforest <- DALEX::model_performance(explainer_randomforest)
resids_xgboost <- DALEX::model_performance(explainer_xgboost)

p1 <- plot(resids_decisiontree, resids_randomforest, resids_xgboost)
p2 <- plot(resids_decisiontree, resids_randomforest, resids_xgboost, geom = "boxplot")
gridExtra::grid.arrange(p1, p2, nrow = 1)

## Variable Importance
vip_decisiontree <- DALEX::variable_importance(explainer_decisiontree, loss_function = DALEX::loss_root_mean_square) 
vip_randomforest <- DALEX::variable_importance(explainer_randomforest, loss_function = DALEX::loss_root_mean_square)
vip_xgboost <- DALEX::variable_importance(explainer_xgboost, loss_function = DALEX::loss_root_mean_square)
plot(vip_decisiontree, vip_randomforest, vip_xgboost, max_vars = 10)

## Break-down Plots for Additive Attributions
new_observation <- custom_new_obs(reg_res_xgboost, pid_train, "diabetes", 1)

## Break-down Plots for Interactions
bd_decisiontree <- breakDown::break_down(explainer_decisiontree, new_observation, keep_distributions = TRUE)
bd_randomforest <- breakDown::break_down(explainer_randomforest, new_observation, keep_distributions = TRUE)
bd_xgboost <- breakDown::break_down(explainer_xgboost, new_observation, keep_distributions = TRUE)

p1_bd <- plot(bd_decisiontree)
p2_bd <- plot(bd_randomforest)
p3_bd <- plot(bd_xgboost)
gridExtra::grid.arrange(p1_bd, p2_bd, p3_bd, nrow = 1)

## Shapley values
shap_new_decisiontree <- DALEX::variable_attribution(explainer_decisiontree, new_observation, type = "shap")
shap_new_randomforest <- DALEX::variable_attribution(explainer_randomforest, new_observation, type = "shap")
shap_new_xgboost <- DALEX::variable_attribution(explainer_xgboost, new_observation, type = "shap")

p1_shap <- plot(shap_new_decisiontree) 
p2_shap <- plot(shap_new_randomforest)
p3_shap <- plot(shap_new_xgboost)
gridExtra::grid.arrange(p1_shap, p2_shap, p3_shap, nrow = 1)

## Local model
local_model_decisiontree <- localModel::individual_surrogate_model(explainer_decisiontree, new_observation, size = 1000, seed = 1313)
local_model_randomforest <- localModel::individual_surrogate_model(explainer_randomforest, new_observation, size = 1000, seed = 1313)
local_model_xgboost <- localModel::individual_surrogate_model(explainer_xgboost, new_observation, size = 1000, seed = 1313)
plot(local_model_decisiontree, local_model_randomforest, local_model_xgboost)

## Ceteris Paribus Explainer
cp_pp_decisiontree <- DALEX::predict_profile(explainer_decisiontree, new_observation)
cp_pp_randomforest <- DALEX::predict_profile(explainer_randomforest, new_observation)
cp_pp_xgboost <- DALEX::predict_profile(explainer_xgboost, new_observation)

p1_cp <- plot(cp_pp_decisiontree, variables = c("age", "glucose"))
p2_cp <- plot(cp_pp_randomforest, variables = c("age", "glucose"))
p3_cp <- plot(cp_pp_xgboost, variables = c("age", "glucose"))

gridExtra::grid.arrange(p1_cp, p2_cp, p3_cp, nrow = 1)

## Dashboard overzicht
modelStudio::modelStudio(explainer_decisiontree)
modelStudio::modelStudio(explainer_randomforest)
modelStudio::modelStudio(explainer_xgboost)
