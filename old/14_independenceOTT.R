# INDEPENDENCE OTT - REWEIGHTING

# 1. Reweighing
balancedData <- function(trainData){
  reweight = NULL
  for (i in 1:nrow(trainData)){
    sex = as.character(trainData[i, "GENDER_NEW"])
    y = as.character(trainData[i, "TARGET"])
    
    sexratio <- nrow(trainData[trainData$GENDER_NEW==sex,])/nrow(trainData)
    yratio <- nrow(trainData[trainData$TARGET==y,])/nrow(trainData)
    actual <- nrow(trainData[trainData$TARGET==y&trainData$GENDER_NEW==sex,])/nrow(trainData)
    reweight <- rbind(reweight, (sexratio*yratio/actual))
  }
  
  return(dplyr::sample_n(trainData, nrow(trainData), replace = T, weight = reweight))
}

