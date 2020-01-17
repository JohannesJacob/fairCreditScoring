### Descriptive analysis - Home Credit Data Set

## setup
# set wd and add libraries
setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/3_code")
path <- "C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/"

library(corrplot)

# read data
df <- read.csv(paste0(path, "2_raw data/home-credit-default-risk/application_train.csv"))
df <- df[!is.na(df$CNT_FAM_MEMBERS),]


## variable examination
# all
c <- sapply(df[2:41], function(x) as.numeric((x)))
mydata.cor = cor(c, method = c("spearman"))
corrplot(mydata.cor, tl.cex = 0.4)

# gender
df <- df[df$CODE_GENDER != 'XNA',]
table(df$TARGET, df$CODE_GENDER)
t.test(df$TARGET~ df$CODE_GENDER)

# age
summary(df$DAYS_BIRTH/365)
df$OLDER_62 <- as.factor(ifelse((df$DAYS_BIRTH/365)>62, '1', '0'))
table(df$TARGET, df$OLDER_62)
t.test(df$TARGET~ df$OLDER_62)

# NA analysis
na_count <-sapply(test, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count$var <- rownames(na_count)
noNA_columns <- rownames(na_count[na_count$na_count==0,])

