# Unterschiede zwischen den Age Groups
setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")


library(ggplot2)

df <- read.csv("2_raw data/taiwan-data/UCI_Credit_Card.csv")

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
ggplot(outlier_free[outlier_free$TARGET=="Bad",], aes(x = CREDIT_AMNT)) +
  geom_density(aes(color = AGE, fill = AGE), alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  xlab("credit debt in TWD")

# Plot
line <- c("groupA", "groupA", "groupA", "groupB", "groupB", "groupB")
x <- c(0, 0.5, 1, 0, 0.5, 1)
y <- c(0, 0.5, 1, 0, 0.5, 1)
df <- as.data.frame(cbind(x,y,line))
ggplot(df, aes(y =  c(0, 0.5, 1), x=  c(0, 0.5, 1))) +
  geom_line() 

##
df$default.payment.next.month <- as.factor(df$default.payment.next.month)
ggplot(df, aes(x = AGE)) +
  geom_density(aes(color = default.payment.next.month, fill = default.payment.next.month), alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))  
