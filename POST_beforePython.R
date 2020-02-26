# POSTPROCESSED DATA - PREDICTIONS

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
dtrain <- read.csv("3_wipResults/taiwan_scaled_train.csv")

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
  "glm"#,
#  "svmRadial", 
#  "rf", 
#  "xgbTree"
#  , "nnet"
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

# Predict on test set: Output = classification & scores
model_prediction <- NULL
cnames <-NULL
for(i in model.names){
  # Define cutoff
  df_cutoff <- NULL
  
  pred <- predict(get(paste("model.", i, sep = "")), newdata = dtest, type = 'prob')$Good
  EMP <- empCreditScoring(scores = pred, classes = dtest$TARGET)
  cutoff <- quantile(pred, EMP$EMPCfrac)
  cutoff_label <- sapply(pred, function(x) ifelse(x>cutoff, 'Good', 'Bad'))

  model_prediction <- cbind(pred, cutoff_label)
  cnames <- c(cnames, c(paste0(i, "_scores"), paste0(i, "_class")))
}
colnames(model_prediction) <- cnames

write.csv(model_prediction, "3_wipResults/POST_Rprediction.csv", row.names = F)

