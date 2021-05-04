# Cross-Validation Functions for Amyloid Models (for origami package)

# Compute Classification Stats ####
classStats <- function(pred, truth, cutoff, lt = FALSE, truth_conv) {
  
  if (lt) {
    pred_vals <- ifelse(pred < cutoff, 1, 0)
  }
  else{
    pred_vals <- ifelse(pred > cutoff, 1, 0)
  }
  
  if (anyNA(pred_vals)){
    na_preds <- is.na(pred_vals)
    
    truth <- truth[!na_preds]
    pred_vals <- pred_vals[!na_preds]
    truth_conv <- truth_conv[!na_preds]
    
  }
  
  p_table <- table(pred_vals, truth)
  
  if (is.null(truth_conv)) {
    conv_corr <- NULL
    pred_pos_conv <- NULL
    pred_neg_conv <- NULL
    pct_pred_pos_conv <- NULL
    pct_pred_neg_conv <- NULL
  }
  else{
    total_pred_pos <- sum(pred_vals)
    total_pred_neg <- length(pred_vals) - total_pred_pos
    conv_corr <- cor(truth_conv, pred_vals)
    pred_pos_conv <- sum(truth_conv[pred_vals == 1])
    pred_neg_conv <- sum(truth_conv[pred_vals == 0])
    pct_pred_pos_conv <- pred_pos_conv/total_pred_pos
    pct_pred_neg_conv <- pred_neg_conv/total_pred_neg
  }
  
  
  tp <- p_table[2,2]/sum(p_table[,2])
  tn <- p_table[1,1]/sum(p_table[,1])
  fp <- p_table[2,1]/sum(p_table[2,])
  fn <- p_table[1,2]/sum(p_table[1,])
  mis_class <- 1 - (sum(diag(p_table))/sum(p_table))
  
  out <- list(mis_class = mis_class,
              true_pos = tp,
              false_pos = fp,
              true_neg = tn,
              false_neg = fn,
              conv_corr = conv_corr,
              pred_pos_conv = pred_pos_conv,
              pred_neg_conv = pred_neg_conv,
              pct_pred_pos_conv = pct_pred_pos_conv,
              pct_pred_neg_conv = pct_pred_neg_conv
              )
  
  return(out)
}

# Linear CV-Function ####
cv_lm <- function(fold, data, reg_form, cutoff) {
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
                        cutoff = cutoff, lt = TRUE, 
                        truth_conv = valid_data$AD_con_any)
  
  # capture results to be returned as output
  out <- list(coef = data.frame(t(coef(mod))),
              SE = ((preds - valid_data[, out_var_ind])^2),
              c_stats = c_stats
  )
  return(out)
}

# Logistic CV-Function ####
cv_logit <- function(fold, data, reg_form, cutoff) {
  # get name and index of outcome variable from regression formula
  out_var <- as.character(unlist(str_split(reg_form, " "))[1])
  out_var_ind <- as.numeric(which(colnames(data) == out_var))
  
  # split up data into training and validation sets
  train_data <- training(data)
  valid_data <- validation(data)
  
  # fit linear model on training set and predict on validation set
  mod <- glm(as.formula(reg_form), data = train_data, family = "binomial")
  preds <- predict(mod, newdata = valid_data, type = "response")
  truth <- valid_data$beta_pos_vote
  
  c_stats <- classStats(pred = preds, truth = truth, 
                        cutoff = cutoff, truth_conv = valid_data$AD_con_any)  
  
  # capture results to be returned as output
  out <- list(coef = data.frame(t(coef(mod))),
              SE = ((preds - valid_data[, out_var_ind])^2),
              c_stats = c_stats
  )
  return(out)
}

# Elastic Net CV-Function ####
cv_net <- function(fold, data, y_var, x_var, cutoff, alpha) {
  # get name and index of outcome variable from regression formula
  #out_var <- as.character(unlist(str_split(reg_form, " "))[1])
  #out_var_ind <- as.numeric(which(colnames(data) == out_var))
  
  train_data <- na.omit(training(data))
  valid_data <- na.omit(validation(data))
  

  # split up data into training and validation sets
  train_y <- as.matrix(train_data[,y_var])
  valid_y <- as.matrix(valid_data[,y_var])
  
  train_x <- as.matrix(train_data[, x_var])
  valid_x <- as.matrix(valid_data[, x_var])
  
  
  # fit linear model on training set and predict on validation set
  mod <- glmnet::glmnet(x = train_x, y = train_y,  family = "binomial", alpha = alpha)
  preds <- as.vector(predict(mod, newx = valid_x, type = "response"))
  truth <- as.vector(valid_y[,1])
  
  c_stats <- classStats(pred = preds, truth = truth, 
                        cutoff = cutoff, truth_conv = valid_data[,"AD_con_any"])  
  
  # capture results to be returned as output
  out <- list(coef = data.frame(t(coef(mod))),
              SE = ((preds - valid_data[, out_var_ind])^2),
              c_stats = c_stats
  )
  return(out)
}

