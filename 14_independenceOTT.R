# INDEPENDENCE OTT - REWEIGHTING

# 1. Reweighing
reweight = NULL
for (i in 1:nrow(dtrain)){
  sex = as.character(dtrain[i, "GENDER_NEW"])
  y = as.character(dtrain[i, "TARGET"])
  
  sexratio <- nrow(dtrain[dtrain$GENDER_NEW==sex,])/nrow(dtrain)
  yratio <- nrow(dtrain[dtrain$TARGET==y,])/nrow(dtrain)
  actual <- nrow(dtrain[dtrain$TARGET==y&dtrain$GENDER_NEW==sex,])/nrow(dtrain)
  reweight <- rbind(reweight, (sexratio*yratio/actual))
}

sample_idx <- dplyr::sample_n(dtrain, nrow(dtrain), replace = T, weight = reweight)

