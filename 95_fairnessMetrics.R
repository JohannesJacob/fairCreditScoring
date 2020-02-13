# Statistical metrics

statParDiff <- function(data = df, sens.attr = "sex", target.attr = "credit"){
  sens.lvls <- levels(data[, sens.attr])
  target.lvls <- levels(data[, target.attr])
  total.count <- nrow(data)
  target1.count <- nrow(data[data[, target.attr]==target.lvls[1],])
  
  p_1 <- (nrow(data[data[, sens.attr] == sens.lvls[1] & data[, target.attr]==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[data[, sens.attr] == sens.lvls[1],])/total.count)
  p_2 <- (nrow(data[data[, sens.attr] == sens.lvls[2] & data[, target.attr]==target.lvls[1],])/target1.count) * 
    (target1.count/total.count) / (nrow(data[data[, sens.attr] == sens.lvls[2],])/total.count)
  return (p_1-p_2)
}