# Write latex output

setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/5_finalResults/")
rm(list = ls());gc()

# Read all files
temp = list.files(pattern="*.csv")
list2env(
  lapply(setNames(temp, make.names(gsub("*.csv$", "", temp))), 
         read.csv), envir = .GlobalEnv)

filenames <- make.names(gsub("*.csv$", "", temp))

latex.output <- NULL
for (f in filenames){
  df <- get(f)
  if (dim(df)[2]>2){
    df = subset(df, !(df[,1] %in% c('EMP','AUC')))

    metric.names <- paste(unique(df[,1]))
    
    avg.metrics <- apply(df[1:4,2:6], 1, function(x){sum(x)/5})
    avg.fairmetrics <- apply(df[5:7,2:6], 1, function(x){sum(abs(x))/5})
    avg.all <- round(c(avg.metrics, avg.fairmetrics),3)
    avg.all <- paste(avg.all, collapse = " & ")
    
    sd.all <- round(apply(df[,2:6], 1, sd),3)
    sd.all <- paste0("(", sd.all, ")", collapse = " & ")
    
    latex.output <- rbind(latex.output, cbind(f, avg.all, sd.all))
  }
}

colnames(latex.output) <- c("file", "average", "sd")
write.csv(latex.output, "latex_output.csv")
