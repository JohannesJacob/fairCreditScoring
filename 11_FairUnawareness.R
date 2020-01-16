# FAIRNESS THROUGH UNAWARENESS

# Omit sensitive attribute and all correelate attributes
# Sensitive attribute is GENDER
sens_attr <- "CODE_GENDER"

# Check correlations
cor.tr_te <- cor(as.matrix(tr_te[, 2]), as.matrix(tr_te[, -2]), use="pairwise.complete.obs", method="spearman") 

cor.tr_te <- rbind(cor.tr_te, abs(cor.tr_te))
cor.tr_te <- cor.tr_te[,order(cor.tr_te[2,], decreasing = T)] 
#hist(cor.tr_te[2,]) #cut-off at 0.1 correlation

# omitting variables
unaware <- t(cor.tr_te)
rownames(unaware) <- colnames(cor.tr_te)
unaware <- unaware[unaware[,2]>0.1&!is.na(unaware[,2]),]
cat(paste(length(rownames(unaware)), "variables have a stronger correlation with GENDER and will be omitted."))

unaware <- c(rownames(unaware), sens_attr)
tr_te <- tr_te[,!(colnames(tr_te) %in% unaware)]

rm(cor.tr_te)