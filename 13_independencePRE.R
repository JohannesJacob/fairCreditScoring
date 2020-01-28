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

# Dummy variables for multinomial variables
for (i in 1:ncol(df)){
  c <- df[,i]
  if (is.factor(c)&length(levels(c))>2){
    dum <- model.matrix(~c)
    colnames(dum) <- paste(names(df)[i], levels(c), sep="_")
  }
}
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
target <- df$TARGET
df <- df[, -which(colnames(df)%in%c("TARGET"))]
fam <- c("multinom", "gaussian", "multinom", "multinom", "gaussian", "multinom", "multinom", )

vec <- 1:ncol(df)
for (i in 1:ncol(df)){
  M <- df
  x <- NULL
  for (j in vec){
    
    y <- colnames(M[j]); if(y=="GENDER_NEW"){next}
    a <- "GENDER_NEW"
    f <- as.formula(paste(y, "~", paste(c(a,x), collapse="+")))
    
    
    if (fam[j]=="multinom") {
      m <- multinom(formula = f, data = M)
      yhat <- predictMNL(m, M)
      t <- unique(M[,j])[match(as.character(yhat), as.character(as.numeric(unique(M[,j]))))]
    } else {
      m <- glm(formula = f, data = M, family = fam[j])
      yhat <- predict(m)
    }
    
    M <- cbind(M, yhat)
    xhat_name <- colnames(M)[ncol(M)] <- paste0(y, "_hat")
    
    if(length(unique(yhat))){next}
    x <- c(x, xhat_name)
  }
  break
  M <- M[22:42]
  collo = NULL; for (f in vec){collo <- c(collo, colnames(df[f]))}
  colnames(M) <- collo
  assign(paste0("M", i), M)
  vec <- c(vec, i); vec <- vec[2:length(vec)]
}

#	2. Iterate through all variables X, so that each variables x is the first predicted variable x' once. Thus, you will generate M data sets (y, x')
#	3. Then, average x' over all M (or build a model for each M and aver-age the prediction y') and predict y'
