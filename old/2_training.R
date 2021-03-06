setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universitšt/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
cat(paste("Run started:", Sys.time(), sep = " "))

#---------------------------
cat("Starting preprocessing...\n")

#source("fairCreditScoring/1_preprocessing.R")
load("3_wipResults/processedData.RData")

packages <- c("caret", "doParallel", "kernlab", "randomForest", "nnet", 
              "xgboost", "foreach", "e1071", "pROC")
sapply(packages, require, character.only = TRUE)
rm(packages)

#---------------------------
# PREPROCESSING
#source("fairCreditScoring/11_FairUnawareness.R")


#---------------------------
# Loop FAIR OTT cost functions
#source cost function !!! in training metric name needs to be changed
#source("assignmentSummaryFUN.R")

#---------------------------
cat("Preparing data...\n")

#Select subset for training and testing
tr_te <- tr_te[tri, ] #kick kaggle test set
trainIndex <- createDataPartition(y, times= 2, p = 0.6, list = FALSE) 
dtrain <- tr_te[trainIndex,]
dval <- tr_te[-trainIndex,]
dt 
ytrain <- factor(y[trainIndex], levels=c("0","1"), labels=c("rejected","accepted"))
yval <- factor(y[-trainIndex], levels=c("0","1"), labels=c("rejected","accepted"))
yt

rm(tr_te, y, tri, trainIndex); gc()

#set trainControl for caret
model.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary, #specific FAIR summary metric
  returnData = FALSE #FALSE, to reduce straining memomry 
)

#---------------------------
cat("Preparing parameter grid for each model...\n")

#Specifications for glm
param.glm <- expand.grid()
args.glm <- list(family = "binomial")

#Specifications for svmRadial (?)
param.svmRadial <- expand.grid(C = c(0.1, 0.5, 1),
                                 sigma = c(0, 0.05, 0.1, 0.15))
args.svmRadial <- list()

#Specifications for rf
param.rf <- expand.grid(mtry = seq(5, 15, by = 1))
args.rf <- list(ntree = 1000)

#Specifications for xgbTree
param.xgbTree <- expand.grid(nrounds = c(100, 200), 
                             max_depth = seq(8, 20, by = 2),
                             gamma = 0, 
                             eta = 0.1, 
                             colsample_bytree = seq(0.6, 1, by = 0.2), 
                             min_child_weight = c(0.5, 1, 3), 
                             subsample = seq(0.4, 0.8, by = 0.2))
args.xgbTree <- list()

#Specifications for nnet           
param.nnet <- expand.grid(decay = seq(0.1, 2, by =  0.1), 
                          size = seq(10, 30, by = 1))
args.nnet <- list(maxit = 100, 
                  trace = FALSE)

#---------------------------
cat("Training model...\n")

#Use parallel computing
nrOfCores  <- detectCores()-1
#registerDoParallel(cores = nrOfCores)
message(paste("\n Registered number of cores for parallel processing:\n",nrOfCores,"\n"))

# Create vector of model names to call parameter grid in for-loop
model.names <- c(#"glm",
                 #"svmRadial", "rf", 
                 "xgbTree"
                 #, "nnet"
                 )

# Train models and save result to model."name"
for(i in model.names) {
  print(i)
  grid <- get(paste("param.", i, sep = ""))
  
  args.train <- list(x = dtrain, 
                     y = ytrain,  
                     method = i, 
                     #tuneGrid  = grid,
                     #metric    = "Accuracy", #needs to be changed depeding on cost function
                     trControl = model.control)
  
  args.model <- c(args.train, get(paste("args.", i, sep = "")))
  
  assign(
    paste("model.", i, sep = ""),
    do.call(train, args.model)
  )
  
  print(paste("Model", i, "finished training:", Sys.time(), sep = " "))
}

for (i in model.names){rm(list=c(paste0('args.',i), paste0('param.',i)))};gc()


#---------------------------
cat("Validating model and setting optimal threshold...\n")

thresholds <- NULL
for(i in model.names){
  model <- get(paste("model.", i, sep = ""))
  
  assign(
    paste("pred.", i, sep = ""),
    predict(model,
            newdata = dval,
            type = "prob")[,2])
  
  result.roc <- roc(yval, get(paste("pred.", i, sep = ""))) # Draw ROC curve.
  result.coords <- coords(result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
  
  
  thresholds <- cbind(thresholds, result.coords[[1]])
  
  print(paste(i, "Validation prediction completed:", Sys.time(), sep = " "))
}

colnames(thresholds) <- c(model.names)
thresholds <- as.data.frame(thresholds)

#---------------------------
cat("Final prediction...\n")

AUCs <- NULL
for(i in model.names){
  model <- get(paste("model.", i, sep = ""))
  model.threshold <- thresholds[[paste0(i)]]
  
  assign(
    paste("pred.", i, sep = ""),
    predict(model,
            newdata = dt,
            type = "prob")[,2])
  model.prediction <- get(paste("pred.", i, sep = ""))
  
  result.roc <- roc(yt, model.prediction) # Draw ROC curve.
  result <- ifelse(model.prediction>model.threshold, 1, 0)
  
  
  AUCs <- cbind(AUCs, auc(yt, result))
  
  print(paste(i, "Testing results completed:", Sys.time(), sep = " "))
}

# Average of AUC
finalAUC <- rowSums(AUCs)/ncol(AUCs)
cat(paste("The final AUC is", finalAUC))
