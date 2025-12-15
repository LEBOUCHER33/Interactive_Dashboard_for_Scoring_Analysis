
# pour déployer on va échantilloner le dataset

import pandas as pd

# 1- on load le dataset
df =  pd.read_csv("./Data/Data_cleaned/application_test_final.csv")

# 2-  on échantillonne
df_sample = df.sample(n=500, random_state=42)

# 3- on crée le csv
df_sample.to_csv("./Scripts/App/assets/data_sample.csv", index=False)

