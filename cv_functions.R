# Cross-Validation Functions for Amyloid Models (for origami package)

# Compute Classification Stats ####
classStats <- function(pred, truth, cutoff, lt = FALSE) {
  
  if (lt) {
    pred_vals <- ifelse(pred < cutoff, 1, 0)
  }
  else{
    pred_vals <- ifelse(pred > cutoff, 1, 0)
  }
  p_table <- table(pred_vals, truth)
  
  tp <- p_table[2,2]/sum(p_table[,2])
  tn <- p_table[1,1]/sum(p_table[,1])
  fp <- p_table[2,1]/sum(p_table[2,])
  fn <- p_table[1,2]/sum(p_table[1,])
  mis_class <- 1 - (sum(diag(p_table))/sum(p_table))
  
  out <- list(mis_class = mis_class,
              true_pos = tp,
              false_pos = fp)
  
  return(out)
}

# Linear CV-Function ####
cv_lm <- function(fold, data, reg_form) {
  # get name and index of outcome variable from regression formula
  out_var <- as.character(unlist(str_split(reg_form, " "))[1])
  out_var_ind <- as.numeric(which(colnames(data) == out_var))
  
  # split up data into training and validation sets
  train_data <- training(data)
  valid_data <- validation(data)
  
  # fit linear model on training set and predict on validation set
  mod <- lm(as.formula(reg_form), data = train_data)
  preds <- predict(mod, newdata = valid_data)
  truth <- valid_data$beta_pos
  c_stats <- classStats(pred = preds, truth = truth,
                        cutoff = 192, lt = TRUE)
  
  # capture results to be returned as output
  out <- list(coef = data.frame(t(coef(mod))),
              SE = ((preds - valid_data[, out_var_ind])^2),
              c_stats = c_stats
  )
  return(out)
}

# Logistic CV-Function ####
cv_logit <- function(fold, data, reg_form) {
  # get name and index of outcome variable from regression formula
  out_var <- as.character(unlist(str_split(reg_form, " "))[1])
  out_var_ind <- as.numeric(which(colnames(data) == out_var))
  
  # split up data into training and validation sets
  train_data <- training(data)
  valid_data <- validation(data)
  
  # fit linear model on training set and predict on validation set
  mod <- glm(as.formula(reg_form), data = train_data, family = "binomial")
  preds <- predict(mod, newdata = valid_data, type = "response")
  truth <- valid_data$beta_pos
  
  c_stats <- classStats(pred = preds, truth = truth, 
                        cutoff = 0.72)  
  
  # capture results to be returned as output
  out <- list(coef = data.frame(t(coef(mod))),
              SE = ((preds - valid_data[, out_var_ind])^2),
              c_stats = c_stats
  )
  return(out)
}