custom_predict <- function(object, newdata, positive_value) { pred <- predict(object, newdata, type = 'prob')
                                              response <- as.vector(pred$.pred_pos)
                                              return(response)}

custom_data_expl <- function(recipe_workflow, dataset, target_variable) { 
                                                                          data_return <- as.data.frame(prep(recipe_workflow$pre$actions$recipe$recipe, dataset) %>% bake(dataset) %>% select(-target_variable))
                                                                          return(data_return)
                                                                        }
custom_y_expl <- function(recipe_workflow, dataset, target_variable) { 
                                                                      data_return <- prep(recipe_workflow$pre$actions$recipe$recipe, dataset) %>% bake(dataset) %>% mutate(target_variable = ifelse(target_variable == 'pos', 1, 0))  %>% pull(target_variable)
                                                                      return(data_return)
                                                                     }
custom_model_expl <- function(recipe_workflow) {return(recipe_workflow$fit$fit)}

custom_new_obs <- function(recipe_workflow, dataset, target_variable, rownumber) {
                              new_obs <- as.data.frame(prep(recipe_workflow$pre$actions$recipe$recipe, dataset) %>% bake(dataset) %>% select(-target_variable))[rownumber,]
                              return(new_obs)
                             }