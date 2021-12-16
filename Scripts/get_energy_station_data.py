#%%
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# %%
date_start = datetime(2015,1,1)
date_end = datetime(2021,12,31)
url = """https://api.energidataservice.dk/datastore_search_sql?sql=SELECT "HourUTC", "GridCompany", "HourlySettledConsumption" FROM  "consumptionpergridarea" WHERE "GridCompany" = '791' AND "HourUTC" >= '""" + date_start.strftime("%Y-%m-%dT%H:%M:%S") + """' AND "HourUTC" < '""" + date_end.strftime("%Y-%m-%dT%H:%M:%S") + """' """

api_call = requests.get(url)
df = pd.DataFrame(api_call.json()["result"]["records"])
print('start')
#extract relevant columns
df.fillna(0, inplace=True)
pkl_name1 = "data/jaegerspris_el_data.pkl"
df.to_pickle(pkl_name1, protocol=3)
print('done')
#%%
#Converting the consumption to means pr week
con = np.array(df_DK1['Con'])
weekly_con = []
iterations = 0
while iterations <= 52:
    hours_in_a_week = 0
    one_week_con = 0
    while hours_in_a_week <= 168:
        hour = (iterations*168)+hours_in_a_week
        one_week_con += con[hour-1]
        hours_in_a_week += 1
    weekly_con.append(one_week_con/7)
    iterations += 1




# %%
plt.plot( df_DK1['HourUTC'],df_DK1['Con'])
plt.show()
# %%
plt.plot(weekly_con)
plt.show()