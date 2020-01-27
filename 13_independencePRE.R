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

# INDEPENDENCE __ PREPROCESSING

#	1. For each x predict x' depending on a, if x is continuous use regression, binary use logistic and for count use poisson. Also integrate the already predicted x'.
target <- df$TARGET
a <- df$GENDER_NEW
df <- df[, -which(colnames(df)%in%c("TARGET", "GENDER_NEW"))]
fam <- c("multinom", "gaussian", "multinom", "multinom", "multinom", ...)
vec <- 1:ncol(df)

for (i in 1:ncol(df)){
  M <- data.frame()
  for (j in vec){
    if (fam[j]=="multinom") {
      m <- multinom(formula = , data = )
    }
    xhat <- predict(m)
    M <- cbind(M, xhat)
  }
  collo = NULL; for (f in vec){collo <- c(collo, colnames(df[f]))}
  colnames(M) <- collo
  assign(paste0("M", i), M)
  vec <- c(vec, i); vec <- vec[2:length(vec)]
}

#	2. Iterate through all variables X, so that each variables x is the first predicted variable x' once. Thus, you will generate M data sets (y, x')
#	3. Then, average x' over all M (or build a model for each M and aver-age the prediction y') and predict y'
