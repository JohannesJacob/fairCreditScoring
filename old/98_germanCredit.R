# Independence Preprocessing

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)

# - read data set (see Fairness Definitions Explained)

df <- read.csv("2_raw data/German credit scoring data/german.data", sep = " ", header = FALSE)
colnames(df) <- c("CHECKING_ACC","DURATION_M", "CREDIT_HIST", "PURPOSE", "AMOUNT", "SAVING_ACC", "EMPLOY_SINCE", "RATE2INCOME", "SEX AND STATUS", "GUARANTORS", 
                  "RESIDENT_SINCE", "PROPERTY", "AGE", "OTHER_INSTALMNT", "HOUSING", "NUMBR_CREDIT", "JOB", "DEPENDENTS", "PHONE", "FOREIGNER", "TARGET")

df$TARGET <- as.factor(ifelse(df$TARGET==1, "GOOD", "BAD"))
df$GENDER_NEW <- as.factor(ifelse(df$`SEX AND STATUS` == 'A92' | df$`SEX AND STATUS` == 'A95', "f", "m"))
df <- df[, -9]
df$SQ_AGE_NEW <- sqrt(df$AGE)

# - training

packages <- c("caret", "doParallel", "kernlab", "randomForest", "nnet", 
              "xgboost", "foreach", "e1071", "pROC")
sapply(packages, require, character.only = TRUE)
rm(packages)

#---------------------------
# PREPROCESSING
source("fairCreditScoring/13_independencePRE.R")

# OTT
source("fairCreditScoring/14_independenceOTT.R")


#---------------------------
cat("Preparing data...\n")

#Select subset for training and testing
trainIndex <- createDataPartition(df$TARGET, p = 0.9, list = FALSE) 
dtrain <- df[trainIndex,] #; dtrain <- balancedData(dtrain) # Independence OTT
dval <- df[-trainIndex,]
rm(trainIndex)

#set trainControl for caret
source("fairCreditScoring/97_germanCreditSummary.R")
model.control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = creditSummary, #specific FAIR summary metric
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
  grid <- get(paste("param.", i, sep = ""))
  
  args.train <- list(TARGET~., 
                     data = dtrain,  
                     method = i, 
                     #tuneGrid  = grid,
                     metric    = "Loss", #needs to be changed depeding on cost function
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
cat("Testing model...\n")

#Print assessment results of level-0-models
test_results <- NULL
obs <- dval$TARGET
for(i in model.names){
  pred <- predict(get(paste("model.", i, sep = "")), newdata = dval)
  test_eval <- rbind(as.numeric(roc(obs, as.numeric(pred))$auc), max(get(paste("model.", i, sep = ""))$results$Loss))
  test_results <- cbind(test_results, test_eval)
}

colnames(test_results) <- model.names; test_results

