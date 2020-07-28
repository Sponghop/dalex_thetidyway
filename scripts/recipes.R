## Data preparatie
data(PimaIndiansDiabetes)
set.seed(1234)
pid_split <- initial_split(PimaIndiansDiabetes, prop = .75)
pid_train <- training(pid_split)
pid_test <- testing(pid_split)

## Preprocessing
mod_rec <-
  recipe(diabetes ~ ., data = pid_train) %>%
  step_mutate(was_pregnant = as.factor(ifelse(pregnant > 0, 'Ja', 'Nee'))) %>%
  step_dummy(was_pregnant) %>%
  step_normalize(all_numeric())