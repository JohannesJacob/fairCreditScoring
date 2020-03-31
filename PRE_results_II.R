# PREPROCESSED DATA PROCESSING

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

# - load packages

packages <- c("caret", "doParallel", "kernlab", "randomForest", "nnet", 
              "xgboost", "foreach", "e1071", "pROC", "EMP", "fairness")
sapply(packages, require, character.only = TRUE)

#Use parallel computing
  # nrOfCores  <- detectCores()-1
  # registerDoParallel(cores = nrOfCores)
  # message(paste("\n Registered number of cores:\n",nrOfCores,"\n"))

rm(packages, nrOfCores)

# - read data set (see Fairness Definitions Explained)

dtest <- read.csv("3_wipResults/taiwan_scaled_test.csv")
dval <- read.csv("3_wipResults/taiwan_scaled_valid.csv")
dtrain <- read.csv("3_wipResults/taiwan_pre_disp_impact_remover.csv") #REWEIGHING

dtest_unscaled <- read.csv("3_wipResults/taiwan_orig_test.csv")
dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT,AGE, TARGET))

# check fairness in training set
source("fairCreditScoring/95_fairnessMetrics.R")

#set trainControl for caret
source("fairCreditScoring/96_empSummary.R")
model.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  verboseIter = T,
  #allowParallel = TRUE,
  summaryFunction = creditSummary, #specific FAIR summary metric
  returnData = FALSE #FALSE, to reduce straining memomry 
)

#---- GRID SEARCH ----

#Specifications for rf
param.rf <- expand.grid(mtry = seq(5, 15, by = 5))
args.rf <- list(ntree = 1000)

#Specifications for nnet           
param.nnet <- expand.grid(decay = seq(0.1, 2.1, by =  1), size = seq(10, 30, by = 10))
args.nnet <- list(maxit = 100, trace = FALSE)

#Specifications for xgbTree
param.xgbTree     <- expand.grid(
  nrounds = c(100, 200),
  max_depth = seq(8, 24, by = 8), 
  gamma = 0,
  eta = 0.1,
  colsample_bytree = seq(0.6, 1, by = 0.3),
  min_child_weight = c(0.5, 1, 3),
  subsample = c(0.4, 0.8)
)

args.xgbTree <- list()

#Specifications for svmLinear      
param.svmLinear <- expand.grid(C = c(0.1, 0.5, 1))
args.svmLinear <- list()

#Specifications for glmnet
param.glm <- NULL
args.glm <- list(family = "binomial")

# Create vector of model names to call parameter grid in for-loop
model.names <- c(
  "glm"
  #,
  #"svmLinear"#, 
  #"rf", 
  #"xgbTree"
  #, "nnet"
)

#---- TRAINING ----

# Train models and save result to model."name"
for(i in model.names) {
  print(i)
  grid <- get(paste("param.", i, sep = ""))
  
  args.train <- list(TARGET~., 
                     data = dtrain,  
                     method = i, 
                     tuneGrid  = grid,
                     metric    = "EMP", #needs to be changed depeding on cost function
                     trControl = model.control)
  
  args.model <- c(args.train
                  , get(paste("args.", i, sep = ""))
                  )
  
  assign(
    paste("model.", i, sep = ""),
    do.call(train, args.model)
  )
  
  print(paste("Model", i, "finished training:", Sys.time(), sep = " "))
}

for (i in model.names){rm(list=c(paste0('args.',i), paste0('param.',i)))};gc()
rm(args.model, args.train, model.control)

#---- THRESHOLDING ----

# Find optimal cutoff based on validation set
for(i in model.names){
  # Define cutoff
  pred <- predict(get(paste("model.", i, sep = "")), newdata = dval, type = 'prob')$Good
  EMP <- empCreditScoring(scores = pred, classes = dval$TARGET)
  assign(paste0('cutoff.', i), quantile(pred, EMP$EMPCfrac))
}

#---- TESTING ----

# Assess test restults
test_results <- NULL

for(i in model.names){

  pred <- predict(get(paste0("model.", i)), newdata = dtest, type = 'prob')$Good
  cutoff <- get(paste0("cutoff.", i))
  cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))

  # Compute AUC
  AUC <- as.numeric(roc(dtest$TARGET, as.numeric(pred))$auc)
  
  # Compute EMP
  EMP <- empCreditScoring(scores = pred, classes = dtest$TARGET)$EMPC
  acceptedLoans <- length(pred[pred>cutoff])/length(pred)

  # Compute Profit from Confusion Matrix (# means in comparison to base scenario = all get loan)
  loanprofit <- NULL
  for (i in 1:nrow(dtest)){
    class_label <- cutoff_label[i]
    true_label <- dtest$TARGET[i]
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
  profitPerLoan <- profit/nrow(dtest)
  
  # fairness criteria average
  statParityDiff <- statParDiff(sens.attr = dtest$AGE, target.attr = cutoff_label)
  
  test_eval <- rbind(AUC, EMP, acceptedLoans, profit, profitPerLoan, statParityDiff)
  test_results <- cbind(test_results, test_eval)
}

# Print results
colnames(test_results) <- c(model.names); test_results

write.csv(test_results, "5_finalResults/PRE_Reweighing_Results.csv", row.names = T)

