# german credit summary

#Cost function for trainControl
creditSummary <- function (data,
                               lev = NULL, 
                               model = NULL) {
  lvls <- levels(data$obs)
  if (length(lvls) > 2) 
    stop(paste("Your outcome has", length(lvls), 
               "levels. The assignmentSummary() function isn't appropriate."))
  if (!all(levels(data[, "pred"]) == lvls)) 
    stop("levels of observed and predicted data do not match")
  #get confusion Matrix
  confMat    <- with(data, table(pred,obs))
  #Calculate profit and losses from the assignment task
  out        <- (confMat[2]*(-5) + confMat[3]*-1)/sum(confMat)
  names(out) <- "Loss"#Metric name
  out
}