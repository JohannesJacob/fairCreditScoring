# INDEPENDENCE/ DEMOGRAPHIC PARITY

library('fairness')

# Preprocessing
data(compas)
dem_parity(data = compas, group ='ethnicity',probs ='probability', preds = NULL,cutoff = 0.4, base ='Caucasian')
dem_parity(data = compas, group ='ethnicity',probs = NULL, preds ='predicted',cutoff = 0.5, base ='Hispanic')
