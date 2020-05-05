# Unterschiede zwischen den Age Groups

library(ggplot2)

dtrain <- read.csv("3_wipResults/taiwan_orig_train.csv")
dtest <- read.csv("3_wipResults/taiwan_orig_test.csv")
dval <- read.csv("3_wipResults/taiwan_orig_valid.csv")
dorig <- rbind(dtrain, dtest, dval)

dorig$AGE <- as.factor(dorig$AGE)
levels(dorig$AGE) <- c("<25", ">=25")

dtrain <- read.csv("3_wipResults/taiwan_scaled_train.csv")
dtest <- read.csv("3_wipResults/taiwan_scaled_test.csv")
dval <- read.csv("3_wipResults/taiwan_scaled_valid.csv")
dscaled <- rbind(dtrain, dtest, dval)

dscaled$AGE <- as.factor(dscaled$AGE)
levels(dscaled$AGE) <- c("<25", ">=25")


# Difference in other categories

# change fill and outline color manually 
ggplot(dscaled, aes(x = LIMIT_BAL)) +
  geom_density(aes(color = AGE, fill = AGE), alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))

outlier_free <-subset(dorig, CREDIT_AMNT<234661.1 & CREDIT_AMNT>-11243.87)
ggplot(outlier_free, aes(x = CREDIT_AMNT)) +
  geom_density(aes(color = TARGET, fill = TARGET), alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  xlab("credit limit")
