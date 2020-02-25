# Statistical metrics

statParDiff <- function(data = df, sens.attr = "AGE", target.attr = "TARGET"){
  sens.var <- as.factor(data[, sens.attr])
  target.var <- as.factor(data[, target.attr])
  sens.lvls <- levels(sens.var)
  target.lvls <- levels(target.var)
  total.count <- nrow(data)
  target1.count <- nrow(data[target.var==target.lvls[1],])
  
  p_1 <- (nrow(data[sens.var == sens.lvls[1] & target.var==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[1],])/total.count)
  p_2 <- (nrow(data[sens.var == sens.lvls[2] & target.var==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[2],])/total.count)
  return (p_1-p_2)
}

avgOddsDiff <- function(data = df, sens.attr = "AGE", target.attr = "TARGET", predicted.attr = "class"){
  data[, sens.attr] <- as.factor(data[, sens.attr])
  
  data_un <- data[data[, sens.attr]==levels(data[, sens.attr])[1],]
  FN_un <- nrow(data_un[data_un[,target.attr] == "Bad" & data_un[,predicted.attr]=="Good",])
  FP_un <- nrow(data_un[data_un[,target.attr] == "Good" & data_un[,predicted.attr]=="Bad",])
  TP_un <- nrow(data_un[data_un[,target.attr] == "Bad" & data_un[,predicted.attr]=="Bad",])
  TN_un <- nrow(data_un[data_un[,target.attr] == "Good" & data_un[,predicted.attr]=="Good",])
  FPR_un <- FP_un/(TN_un+FP_un)
  TPR_un <- TP_un/(TP_un+FN_un)
  
  data_priv <- data[data[, sens.attr]==levels(data[, sens.attr])[2],]
  FN_priv <- nrow(data_priv[data_priv[,target.attr] == "Bad" & data_priv[,predicted.attr]=="Good",])
  FP_priv <- nrow(data_priv[data_priv[,target.attr] == "Good" & data_priv[,predicted.attr]=="Bad",])
  TP_priv <- nrow(data_priv[data_priv[,target.attr] == "Bad" & data_priv[,predicted.attr]=="Bad",])
  TN_priv <- nrow(data_priv[data_priv[,target.attr] == "Good" & data_priv[,predicted.attr]=="Good",])
  FPR_priv <- FP_priv/(TN_priv+FP_priv)
  TPR_priv <- TP_priv/(TP_priv+FN_priv)
  
  
  return (((FPR_un-FPR_priv)+(TPR_un-TPR_priv))/2)
}
