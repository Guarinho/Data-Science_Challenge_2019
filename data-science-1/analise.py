import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
#import seaborn as sns
#from statsmodels.distributions.empirical_distribution import ECDF
from sklearn import preprocessing
from scipy.stats import norm


np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})

normal = df.loc[:,"normal"]
binomial = df.loc[:,"binomial"]

m_norm = normal.describe().mean()
v_norm = normal.describe().std()

m_binom = binomial.describe().mean()
v_binom = binomial.describe().std()

resultado = (round(m_binom - m_norm, 3), round(v_binom - v_norm,3))


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)

mean_profile = stars.loc[stars["target"] > 0]["mean_profile"]


x = mean_profile.values.reshape(-1, 1)
min_max_scaler = preprocessing.StandardScaler()
x_scaled = min_max_scaler.fit_transform(x)
false_pulsar_mean_profile_standardized = pd.DataFrame(x_scaled)
print(false_pulsar_mean_profile_standardized.describe())


oi = norm.ppf(0.80, loc=0, scale=1)
nove = norm.ppf(0.90, loc=0, scale=1)
noc = norm.ppf(0.95, loc=0, scale=1)
print( oi, nove, noc)

df_f = false_pulsar_mean_profile_standardized.describe()
normal_25 = df_f.loc["25%"] - norm.ppf(0.25, loc=0, scale=1)
normal_50 = df_f.loc["50%"] - norm.ppf(0.50, loc=0, scale=1)
normal_75 = df_f.loc["75%"] - norm.ppf(0.75, loc=0, scale=1)


normal = df.loc[:,"normal"]
desvio = normal.describe().std()
media = normal.describe().mean()
pro1 = norm.cdf(media-desvio)
pro2 = norm.cdf(media+desvio)

mean_profile = stars["mean_profile"][stars["target"] == False]
x = mean_profile.values.reshape(-1, 1)
min_max_scaler = preprocessing.StandardScaler()
x_scaled = min_max_scaler.fit_transform(x)
false_pulsar_mean_profile_standardized = pd.DataFrame(x_scaled)
ppf = sct.norm.ppf([0.25, 0.5, 0.75])
normal_25 = false_pulsar_mean_profile_standardized.describe().loc['25%']
normal_50 = false_pulsar_mean_profile_standardized.describe().loc['50%']
normal_75 = false_pulsar_mean_profile_standardized.describe().loc['75%']
print( ((normal_25-ppf[0]).round(decimals=3), (normal_50-ppf[1]).round(decimals=3), (normal_75-ppf[2]).round(decimals=3)))


df_f = stars['mean_profile'][stars['target'] == False]
false_pulsar_mean_profile_standardized = (df_f - df_f.mean())/df_f.std(ddof=0)
ppf = sct.norm.ppf([0.25, 0.5, 0.75])
q1 = false_pulsar_mean_profile_standardized.describe()['25%']
q2 = false_pulsar_mean_profile_standardized.describe()['50%']
q3 = false_pulsar_mean_profile_standardized.describe()['75%']

print(((q1-ppf[0]).round(decimals=3), (q2-ppf[1]).round(decimals=3), (q3-ppf[2]).round(decimals=3)))