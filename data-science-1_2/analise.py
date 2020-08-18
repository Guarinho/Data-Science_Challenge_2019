import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
#import seaborn as sns
#from statsmodels.distributions.empirical_distribution import ECDF





np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})



normal = df.loc[:,"normal"]
binomial = df.loc[:,"binomial"]


q1_norm = normal.describe().loc["25%"]
q1_binom = binomial.describe().loc["25%"]

q2_norm = normal.describe().loc["50%"]
q2_binom = binomial.describe().loc["50%"]

q2_norm = normal.describe().loc["75%"]
q2_binom = binomial.describe().loc["75%"]

resultado = (q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)