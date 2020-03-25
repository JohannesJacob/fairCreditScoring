# POSTPROCESSING PROFIT EVALUATION

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

thresholds <- read.csv("3_wipResults/ROC_POST_thresholds.csv")
model.names <- c('glm', "svmLinear", "rf", "xgbTree", "nnet")

#Print test results
test_results <- NULL
obs <- dtest_unscaled$TARGET
for(m in model.names){
  score <- thresholds[,paste0(m,"_fairScores")]
  class <- thresholds[,paste0(m,"_fairLabels")]
  
  # Compute AUC
  AUC <- as.numeric(roc(obs, as.numeric(score))$auc)
  
  # Compute EMP
  acceptedLoans <- length(class[class=="Good"])/length(class)

  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(dtest_unscaled)){
    class_label <- class[i]
    true_label <- dtest_unscaled$TARGET[i]
    if (class_label == "Bad" & true_label == "Bad"){
      #p = dtest_unscaled$CREDIT_AMNT[i]
      p = 0
    } else if (class_label == "Good" & true_label == "Bad"){
      p = -dtest_unscaled$CREDIT_AMNT[i] 
    } else if (class_label == "Good" & true_label == "Good"){
      p = dtest_unscaled$CREDIT_AMNT[i] * 0.2644
    }else if (class_label == "Bad" & true_label == "Good"){
      p = -dtest_unscaled$CREDIT_AMNT[i] * 0.2644
      #p = 0
    }
    loanprofit <- c(loanprofit, p)
  }
  profit <- sum(loanprofit)
  profitPerLoan <- profit/nrow(dtest_unscaled)
  
  # fairness criteria average
  dtest_unscaled$CLASS <- class
  statParityDiff <- statParDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'CLASS')
  averageOddsDiff <- avgOddsDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET', predicted.attr = 'CLASS')
  
  test_eval <- rbind(AUC, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff)
  test_results <- cbind(test_results, test_eval)
}

# Add base scenario = all get loan
AUC <- as.numeric(roc(obs, rep.int(2, nrow(dtest_unscaled)))$auc)
acceptedLoans <- 1
loanprofit <- NULL
for (i in 1:nrow(dtest_unscaled)){
  p = ifelse(dtest_unscaled$TARGET[i]=="Bad", -dtest_unscaled$CREDIT_AMNT[i], dtest_unscaled$CREDIT_AMNT[i] * 0.2644)
  loanprofit <- c(loanprofit, p)
}
profit <- sum(loanprofit)
profitPerLoan <- profit/nrow(dtest_unscaled)

statParityDiff <- statParDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET')
dtest_unscaled$BASE_PRED <- factor(rep("Good", nrow(dtest_unscaled)), levels = c("Bad", "Good"))
averageOddsDiff <- avgOddsDiff(data = dtest_unscaled, sens.attr = 'AGE', target.attr = 'TARGET', predicted.attr = 'BASE_PRED')


test_eval <- rbind(AUC, acceptedLoans, profit, profitPerLoan, statParityDiff, averageOddsDiff)
test_results <- cbind(test_eval, test_results)

rm(acceptedLoans, AUC, class_label, cutoff, cutoff_label, EMP, i, loanprofit, obs, p,  
   profit, profitPerLoan, statParityDiff, true_label, test_eval)

# Print results
colnames(test_results) <- c("base", model.names); test_results

write.csv(test_results, "5_finalResults/POST_ROC_results.csv", row.names = T)
