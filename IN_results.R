# INPROCESSING DATA MODEL SELECTION

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# libraries
library(EMP)
library(pROC)
source("fairCreditScoring/95_fairnessMetrics.R")

# read data
dtest_unscaled <- read.csv("3_wipResults/taiwan_orig_test.csv")

prPredictions <- read.csv("3_wipResults/pr_predictions.csv")
metaPredictions <- read.csv("3_wipResults/meta_predictions_tau_06.csv")
model.names <- c("prPredictions") #, "metaPredictions")


#Print test results
test_results <- NULL
obs <- dtest_unscaled$TARGET
for(m in model.names){
  # get model
  model <- get(m)
  
  # calculate emp values
  empVals <- NULL
  for (c in 1:ncol(model)){
    empVal <- empCreditScoring(model[,c], dtest_unscaled$TARGET)
    empVals <- unlist(c(empVals, empVal["EMPC"]))
  }
  bestPrediction <- model[, which(empVals == max(empVals))]
  
  # Define cutoff
  df_cutoff <- NULL

  # Define cutoff
  EMP <- empCreditScoring(scores = bestPrediction, classes = obs)
  cutoff <- EMP$EMPCfrac
  cutoff <- quantile(pred, EMP$EMPCfrac)
  cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))
  df_cutoff <- cbind(dtest, CLASS = cutoff_label)
  
  # Compute AUC
  AUC <- as.numeric(roc(obs, as.numeric(bestPrediction))$auc)
  
  # Compute EMP
  acceptedLoans <- EMP$EMPCfrac
  EMP <- EMP$EMPC
  
  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(df_cutoff)){
    class_label <- df_cutoff$CLASS[i]
    true_label <- df_cutoff$TARGET[i]
    if (class_label == "Bad" & true_label == "Bad"){
      #p = df_cutoff$CREDIT_AMNT[i]
      p = 0
    } else if (class_label == "Good" & true_label == "Bad"){
      p = -df_cutoff$CREDIT_AMNT[i] 
    } else if (class_label == "Good" & true_label == "Good"){
      p = df_cutoff$CREDIT_AMNT[i] * 0.2644
    }else if (class_label == "Bad" & true_label == "Good"){
      p = -df_cutoff$CREDIT_AMNT[i] * 0.2644
      #p = 0
    }
    loanprofit <- c(loanprofit, p)
  }
  profit <- sum(loanprofit)
  profitPerLoan <- profit/nrow(dtest_unscaled)
  
  # fairness criteria average
  statParityDiff <- statParDiff(data = df_cutoff, sens.attr = 'AGE', target.attr = 'CLASS')
  averageOddsDiff <- avgOddsDiff(data = df_cutoff, sens.attr = 'AGE', target.attr = 'TARGET', predicted.attr = 'CLASS')
  
  test_eval <- rbind(AUC, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff)
  test_results <- cbind(test_results, test_eval)
}

# Add base scenario = all get loan
AUC <- as.numeric(roc(obs, rep.int(2, nrow(dtest_unscaled)))$auc)
EMP <- NA
acceptedLoans <- 1
loanprofit <- NULL
for (i in 1:nrow(dtest_unscaled)){
  p = ifelse(dtest_unscaled$TARGET[i]=="Bad", -dtest_unscaled$CREDIT_AMNT[i], df_cutoff$CREDIT_AMNT[i] * 0.2644)
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest_unscaled)

statParityDiff <- statParDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET')
dtest_unscaled$BASE_PRED <- factor(rep("Good", nrow(dtest_unscaled)), levels = c("Bad", "Good"))
averageOddsDiff <- avgOddsDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET', predicted.attr = 'BASE_PRED')


test_eval <- rbind(AUC, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff)
test_results <- cbind(test_eval, test_results)

rm(acceptedLoans, AUC, class_label, cutoff, cutoff_label, EMP, i, loanprofit, obs, p,  
   profit, profitPerLoan, statParityDiff, true_label, test_eval)

# Print results
colnames(test_results) <- c("base", model.names); test_results
