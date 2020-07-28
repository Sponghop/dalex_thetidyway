source("scripts/tidymodels.R")
source("scripts/custom_functions.R")

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