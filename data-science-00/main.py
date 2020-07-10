#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[3]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[1]:


def q1():
     return (black_friday.shape)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
  Age = black_friday.groupby(["Age","Gender"]).count()
  A26_35 = Age.loc["26-35","User_ID"]["F"]
  return int(A26_35)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    Total = black_friday.groupby(["User_ID"]).nunique().sum()
    Tolal_us = Total["User_ID"]
    return int(Tolal_us)


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    dados = len(black_friday.dtypes.value_counts())
    return dados


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    nul_total = black_friday.isna().sum().max()/len(black_friday)
    return float(nul_total)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    headers = list(black_friday.columns.values)
    nun_col = black_friday[headers].isnull().sum().max()
    return int(nun_col)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    fre_grp3 = black_friday["Product_Category_3"].value_counts().idxmax()
    return fre_grp3


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    data = black_friday['Purchase']
    dataf=((data-data.min())/(data.max()-data.min()))
    return float(dataf.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    norm=(((black_friday['Purchase']-black_friday['Purchase'].mean())/black_friday['Purchase'].std()))
    oc = norm.between(-1, 1, inclusive=True).value_counts().loc[True]
    return float(oc)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    conditions = [(black_friday['Product_Category_2'].isna() ) & (black_friday['Product_Category_3'].isna())]
    choices = [1]
    black_friday['Logica'] = np.select(conditions, choices)
    condi =  bool(black_friday['Logica'].sum())
    return condi

