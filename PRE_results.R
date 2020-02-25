# PREPROCESSED DATA PROCESSING

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# - load packages

packages <- c("caret", "doParallel", "kernlab", "randomForest", "nnet", 
              "xgboost", "foreach", "e1071", "pROC", "EMP", "fairness")
sapply(packages, require, character.only = TRUE)
rm(packages)

# - read data set (see Fairness Definitions Explained)

dtest <- read.csv("3_wipResults/taiwan_scaled_test.csv")
dtrain <- read.csv("3_wipResults/taiwan_pre_reweighing.csv")

dtest_unscaled <- read.csv("3_wipResults/taiwan_orig_test.csv")
dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT,AGE, TARGET))

# check fairness in training set
source("fairCreditScoring/95_fairnessMetrics.R")
statParDiff(data = dtrain, sens.attr = 'AGE', target.attr = 'TARGET')

#set trainControl for caret
source("fairCreditScoring/96_empSummary.R")
model.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  verboseIter = T,
  summaryFunction = creditSummary, #specific FAIR summary metric
  returnData = FALSE #FALSE, to reduce straining memomry 
)

# TO DO: Define tuning parameters

# Create vector of model names to call parameter grid in for-loop
model.names <- c(
  "glm",
  "svmRadial", 
  "rf", 
  "xgbTree"
  , "nnet"
)

# Train models and save result to model."name"
for(i in model.names) {
  print(i)
  #grid <- get(paste("param.", i, sep = ""))
  
  args.train <- list(TARGET~., 
                     data = dtrain,  
                     method = i, 
                     #tuneGrid  = grid,
                     metric    = "EMP", #needs to be changed depeding on cost function
                     trControl = model.control)
  
  args.model <- c(args.train
                  #, get(paste("args.", i, sep = ""))
                  )
  
  assign(
    paste("model.", i, sep = ""),
    do.call(train, args.model)
  )
  
  print(paste("Model", i, "finished training:", Sys.time(), sep = " "))
}

for (i in model.names){rm(list=c(paste0('args.',i), paste0('param.',i)))};gc()
rm(args.model, args.train, model.control)

#---------------------------
cat("Testing model...\n")

#Print assessment results of level-0-models
test_results <- NULL
obs <- dtest$TARGET
for(i in model.names){
  # Define cutoff
  df_cutoff <- NULL
  
  pred <- predict(get(paste("model.", i, sep = "")), newdata = dtest, type = 'prob')$Good
  EMP <- empCreditScoring(scores = pred, classes = obs)
  cutoff <- quantile(pred, EMP$EMPCfrac)
  cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))
  df_cutoff <- cbind(dtest, CLASS = cutoff_label)
  
  # Compute AUC
  AUC <- as.numeric(roc(obs, as.numeric(pred))$auc)
  
  # Compute EMP
  acceptedLoans <- EMP$EMPCfrac
  EMP <- EMP$EMPC

  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(df_cutoff)){
    class_label <- df_cutoff$CLASS[i]
    true_label <- df_cutoff$TARGET[i]
    if (class_label == "Bad" & true_label == "Bad"){
      #p = dtest_unscaled$CREDIT_AMNT[i]
      p = 0
    } else if (class_label == "Good" & true_label == "Bad"){
      p = -dtest_unscaled$CREDIT_AMNT[i] 
    } else if (class_label == "Good" & true_label == "Good"){
      p = dtest_unscaled$CREDIT_AMNT[i] * 0.2644
    }else if (class_label == "Bad" & true_label == "Good"){
      #p = -dtest_unscaled$CREDIT_AMNT[i] * 1.2644
      p = 0
    }
    loanprofit <- c(loanprofit, p)
  }
  profit <- sum(loanprofit)
  profitPerLoan <- profit/nrow(dtest)
  
  # fairness criteria average
  statParityDiff <- statParDiff(data = df_cutoff, sens.attr = 'AGE', target.attr = 'CLASS')
  
  test_eval <- rbind(AUC, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff)
  test_results <- cbind(test_results, test_eval)
}

# Add base scenario = all get loan
AUC <- as.numeric(roc(obs, rep.int(1, nrow(dtest_unscaled)))$auc)
EMP <- NA
acceptedLoans <- 1
loanprofit <- NULL
for (i in 1:nrow(dtest_unscaled)){
  p = ifelse(dtest_unscaled$TARGET[i]=="Bad", -dtest_unscaled$CREDIT_AMNT[i], dtest_unscaled$CREDIT_AMNT[i] * 0.2644)
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest_unscaled)

statParityDiff <- statParDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET')

test_eval <- rbind(AUC, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff)
test_results <- cbind(test_eval, test_results)

rm(acceptedLoans, AUC, class_label, cutoff, cutoff_label, EMP, i, loanprofit, obs, p, pred, 
   profit, profitPerLoan, statParityDiff, true_label, test_eval)

# Print results
colnames(test_results) <- c("base", model.names); test_results


