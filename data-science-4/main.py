#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import( TfidfVectorizer, CountVectorizer)


# In[2]:


## Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


df = pd.read_csv("countries.csv", sep=",", header=0, names=[
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"], index_col=None, usecols=[
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"], squeeze=False, mangle_dupe_cols=True, decimal=',')
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df


# In[ ]:





# In[ ]:





# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[ ]:





# In[7]:


df[["Country","Region"]].apply(lambda x: x.str.strip())
#df.apply(lambda x: x.replace(',', '.')).astype(float)


# In[8]:


df


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    columns = ["Country","Region"]
    df[columns] = df[columns].apply(lambda x: x.str.strip())
    
    return sorted(df.Region.unique())


# In[10]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[11]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    dis = discretizer.fit_transform(df[["Pop_density"]])

    q90 = np.quantile(dis, 0.9)
    
    return int(sum(dis> q90))
    


# In[ ]:





# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[12]:


def q3():
     #one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
     #course_encoded = one_hot_encoder.fit_transform(df[["Region","Climate"]])
     encoded_columns_1 = pd.get_dummies(data=df,drop_first = False,prefix_sep='__', columns=['Region','Climate'])
     return int(encoded_columns_1.filter(regex = '__').shape[1] +1)
        
    


# In[ ]:





# In[ ]:





# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[13]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[14]:


def q4():
    df_pipe = df.copy()
    pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ('standard_scaler', StandardScaler())
    ])
    #pipe.fit(df._get_numeric_data())
    pipe.fit_transform(df.select_dtypes(include=np.number))
    #df_to_transform = df.dtypes[(df.dtypes == "int64")
    #                |(df.dtypes == "float64")].index.to_list()
    return float(pipe.transform([test_country[2:]])[0][9].round(3))


# In[15]:


#sk.utils.all_estimators()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[16]:


def q5():
    iqr = df.Net_migration.quantile(0.75) - df.Net_migration.quantile(0.25)

    lim_inf = df.Net_migration.quantile(0.25) - 1.5*iqr
    lim_sup = df.Net_migration.quantile(0.75) + 1.5*iqr
    o=df["Net_migration"].value_counts().sum()
    out_inf = (df.Net_migration < lim_inf).sum()
    out_sup = (df.Net_migration > lim_sup).sum()
    return (int(out_inf),int(out_sup),False)


# In[17]:



sns.set(style="whitegrid")

ax = sns.boxplot(x=df["Net_migration"])


# In[ ]:





# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[18]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
dts = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[19]:


def q6():
    vectorizer = CountVectorizer()
    return int(vectorizer.fit_transform(dts.data)[:,vectorizer.vocabulary_['phone']].sum())


# In[ ]:





# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[20]:


def q7():
    tfvtz = TfidfVectorizer()
    return float(tfvtz.fit_transform(dts.data)[:,tfvtz.vocabulary_['phone']].sum().round(3))
    


# In[ ]:





# In[ ]:




