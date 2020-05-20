# result statistics

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

library(ggplot2)

# pairwise p-test
results <- read.csv("5_finalResults/ALL_RESULTS.csv")
results <- results[results$Statistic =="Mean",]
with(results, pairwise.t.test(Profit, ï..Processor))


# Plot
results <- read.csv("5_finalResults/ALL_RESULTS_transposed.csv")
results <- results[results$ï..Processor!="Base  Model",]
colnames(results)[1] <- "Fair_Processor"
results$Fair_Processor <- factor(results$Fair_Processor, levels = c("Max Profit", "Reweighing", "DI Remover", "PR Remover",
                                                                    "Reject Opt", "Adv. Debiasing", "Equ. Odds", "Meta Fair Classif.",
                                                                    "Group Platt Scaling"), )

cols <- c("BalAcc", "ProfitpLoan", "IND", "SEP", "SUF")
for (cn in cols){
  results[paste0(cn, "_se")] <- results[cn]/sqrt(5)
}

# BalAcc
pd <- position_dodge(0.1)
ggplot(results, aes(x= Fair_Processor, y=BalAcc)) + 
  geom_errorbar(aes(ymin=BalAcc-BalAcc_STD, ymax=BalAcc+BalAcc_STD), colour="black", width=.1) +
  geom_point(position=pd, size=3)

# ProfitpLoan
pd <- position_dodge(0.1)
ggplot(results, aes(x= Fair_Processor, y=ProfitpLoan)) + 
  geom_errorbar(aes(ymin=ProfitpLoan-ProfitpLoan_STD, ymax=ProfitpLoan+ProfitpLoan_STD), colour="black", width=.1) +
  geom_point(position=pd, size=3)

# Fairness Criteria
fairCrit <- read.csv("5_finalResults/ALL_RESULTS_fairCriteria.csv")
colnames(fairCrit)[1] <- "Fair_Processor"
results$Fair_Processor <- factor(results$Fair_Processor, levels = c("Max Profit", "Reweighing", "DI Remover", "PR Remover",
                                                                    "Reject Opt", "Adv. Debiasing", "Equ. Odds", "Meta Fair Classif.",
                                                                    "Group Platt Scaling"), )

ggplot(fairCrit, aes(x=Fair_Processor, y=Mean, colour=Criteria, group=Criteria)) + 
  geom_errorbar(aes(ymin=Mean-STD, ymax=Mean+STD), colour="black", width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd, size=3)
