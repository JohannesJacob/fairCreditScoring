# Check age cutoff with biggest disparity

df <- read.csv(file = "2_raw data/taiwan-data/UCI_Credit_Card.csv")
colnames(df)[which(names(df) == "default.payment.next.month")] <- "TARGET" #Attention is the opposite of Target

disparity <- NULL
for (cutoff in unique(df$AGE)){
  df$AGE_cut <- sapply(df$AGE, function(x) ifelse(x<cutoff, 'YOUNG', 'AGED'))
  d <- statParDiff(df, 'AGE_cut', 'TARGET')
  names(d) <- paste0('cut', cutoff)
  disparity <- c(disparity, d)
  
  df <- subset(df, select = -AGE_cut)
}
View(disparity[order(disparity)])