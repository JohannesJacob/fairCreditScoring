# INDEPENDENCE/ DEMOGRAPHIC PARITY

library('fairness')

# Preprocessing

# OTT

# Postprocessing based on final classification (not their propabilities)
data(compas)
indep <- dem_parity(data = compas, group ='ethnicity',probs = NULL, preds ='predicted',cutoff = 0.5, base ='Caucasian')


while (condition) {
  
}
