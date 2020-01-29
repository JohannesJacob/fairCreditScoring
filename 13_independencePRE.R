# - read data set (see Fairness Definitions Explained)

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)

df <- read.csv("2_raw data/German credit scoring data/german.data", sep = " ", header = FALSE)
colnames(df) <- c("CHECKING_ACC","DURATION_M", "CREDIT_HIST", "PURPOSE", "AMOUNT", "SAVING_ACC", "EMPLOY_SINCE", "RATE2INCOME", "SEX AND STATUS", "GUARANTORS", 
                  "RESIDENT_SINCE", "PROPERTY", "AGE", "OTHER_INSTALMNT", "HOUSING", "NUMBR_CREDIT", "JOB", "DEPENDENTS", "PHONE", "FOREIGNER", "TARGET")

df$TARGET <- as.factor(ifelse(df$TARGET==1, "GOOD", "BAD"))
df$GENDER_NEW <- as.factor(ifelse(df$`SEX AND STATUS` == 'A92' | df$`SEX AND STATUS` == 'A95', "f", "m"))
df <- df[, -9]
df$SQ_AGE_NEW <- sqrt(df$AGE)

# - mulinomial predict function
predictMNL <- function(model, newdata) {
  
  # Only works for neural network models
  if (is.element("nnet",class(model))) {
    # Calculate the individual and cumulative probabilities
    probs <- predict(model,newdata,"probs")
    cum.probs <- t(apply(probs,1,cumsum))
    
    # Draw random values
    vals <- runif(nrow(newdata))
    
    # Join cumulative probabilities and random draws
    tmp <- cbind(cum.probs,vals)
    
    # For each row, get choice index.
    k <- ncol(probs)
    ids <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
    
    # Return the values
    return(ids)
  }
}


# INDEPENDENCE __ PREPROCESSING
library(nnet)

#	1. For each x predict x' depending on a, if x is continuous use regression, binary use logistic and for count use poisson. Also integrate the already predicted x'.
#	2. Iterate through all variables X, so that each variables x is the first predicted variable x' once. Thus, you will generate M data sets (y, x')

target <- df$TARGET
df <- df[, -which(colnames(df)%in%c("TARGET"))]
fam <- c("multinom", "gaussian", "multinom", "multinom", "gaussian", "multinom", "multinom", "gaussian", "multinom", "gaussian",
         "multinom", "gaussian", "multinom", "multinom", "poisson", "multinom", "poisson", "binomial", "binomial", "binomial", 
         "gaussian")

vec <- c(1:19, 21) #exclude gender
for (i in c(1:19, 21)){
  M <- df
  x <- NULL
  print(vec[1])
  for (j in vec){
    
    y <- colnames(M[j])
    a <- "GENDER_NEW"
    f <- as.formula(paste(y, "~", paste(c(a,x), collapse="+")))
    
    
    if (fam[j]=="multinom") {
      m <- multinom(formula = f, data = M, trace=F)
      yhat <- predictMNL(m, M)
      yhat <- as.numeric(unique(M[,j])[match(as.character(yhat), as.character(as.numeric(unique(M[,j]))))])
    } else if (fam[j]=="binomial"){
      m <- glm(formula = f, data = M, family = fam[j])
      yhat <- predict(m, type = "response")
    } else {
      m <- glm(formula = f, data = M, family = fam[j])
      yhat <- predict(m)
    }
    
    M <- cbind(M, yhat)
    xhat_name <- colnames(M)[ncol(M)] <- y
    
    if(length(unique(yhat))){next}
    x <- c(x, xhat_name)
  }
  M <- M[, 22:41]
  M <- M[, order(colnames(M))]
  assign(paste0("M", i), M)
  vec <- c(vec, i); vec <- vec[2:length(vec)]
}
rm(a, f, fam, i, j, vec, x, xhat_name, y, yhat, M, m)
#	3. Then, average x' over all M (or build a model for each M and aver-age the prediction y') and predict y'
dfs <- list(M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M21)
for (i in c(1:19, 21)){rm(list=c(paste0('M',i)))};gc()

M_mean <- NULL
for (c in 1:20){
  res <- rowMeans(sapply(dfs,function(x){return(x[,c])}))
  M_mean <- cbind(M_mean, res)
}
M_mean <- as.data.frame(M_mean)
colnames(M_mean) <- colnames(dfs[[1]])
rm(dfs, c, res, i)

# OUTPUT

M_mean$TARGET <- target

df <- M_mean; rm(M_mean)

# - IF I LEAVE IT LIKE THIS, TEST DATA NEEDS TO BE TRANSFORMED AS WELL

# Average factors
#cols <- colnames(M_mean)
#for (c in cols){
#  if (class(df[,c])=="factor"){
#    cutoff <- runif(nrow(M_mean), min = 1, max = ceiling(M_mean[,c]))
#    tmp <- cbind(M_mean[,c], cutoff)
#    r <- apply(tmp, 1, function(x){
#      if(tmp[,1]<tmp[,2]){floor(x)} else {ceiling(x)}
#    })
#    M_mean[,c] <- r
#  }
#}

