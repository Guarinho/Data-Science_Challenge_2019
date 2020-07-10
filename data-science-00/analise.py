# with open("black_friday.csv","r" ) as file:

#     data = file.read()

# print(data)

import pandas as pd
import numpy as np
from sklearn import preprocessing



black_friday = pd.read_csv("black_friday.csv")



info = black_friday.info()
describe = black_friday.describe(include="all")
Na = black_friday.isna()
shap = black_friday.shape
typ = black_friday.dtypes


print(info) # typ, shap
print(describe) # estatistic
print(Na) # Na boolean answer
print(shap) # n x m
print(typ) # objects , float , int


Age = black_friday.groupby(["Age","Gender"]).count()
print('****aaaaaaaaaaaaaaaaaaaa*****')
p=black_friday.empty
print(p)
print('****aaaaaaaaaaaaaaaaaaaa*****')

A26_35 = Age.loc["26-35","User_ID"]["F"]
filtro =['Gender','Age']


print(A26_35)
print('*********')

    
    
    
    

Total = black_friday.groupby(["User_ID"]).nunique().sum()
Tolal_us = Total["User_ID"]
print(Tolal_us)

# Total_re = black_friday.count().sum()
# print(Total_re)
# Na_re = black_friday.isna()



headers = list(black_friday.columns.values)
nun_col = black_friday[headers].isnull().sum().max()
print(nun_col)

print("4")
dados = len(black_friday.dtypes.value_counts())
print(dados)





nul_total = black_friday.isna().sum().max()/len(black_friday)
print(nul_total)














maior_nul= black_friday.isna().sum().max()
print(maior_nul)



fre_grp3 = black_friday["Product_Category_3"].value_counts().idxmax()
print(fre_grp3 )


df = black_friday[["Purchase"]].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
print(df_normalized.mean())

norm=(((black_friday['Purchase']-black_friday['Purchase'].mean())/black_friday['Purchase'].std()))
oc = norm.between(-1, 1, inclusive=True).value_counts().loc[True]
print(oc)



conditions = [
    (black_friday['Product_Category_2'].isna()) & (black_friday['Product_Category_3'].notna())
     ]
#choices = [1]
#print(conditions)
#black_friday['Logica'] = np.select(conditions, choices)
condi =  bool(~sum(conditions).any())
print(condi)
#Condition_one = black_friday['Product_Category_2'].where(black_friday['Product_Category_2'].notna().all() == black_friday['Product_Category_3'].notna().all())
#Condition_one.all()





#age1= Age.convert_dtypes()
#typ1 = age1.dtypes
#print(age1)
#print(typ1) # objects , float , int
