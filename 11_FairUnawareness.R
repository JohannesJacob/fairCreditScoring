# FAIRNESS THROUGH UNAWARENESS

# Omit sensitive attribute and all correelate attributes
# Sensitive attribute is GENDER

# Check correlations
cor.tr_te <- cor(as.matrix(tr_te[, 2]), as.matrix(tr_te[, -2]), use="pairwise.complete.obs", method="spearman") 

