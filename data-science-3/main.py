#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression)
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from loguru import logger
#from sklearn.model_selection import StratifiedKFold
#from yellowbrick.datasets import load_credit
#from yellowbrick.features import PCA

#from sklearn.datasets import load_digits


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


fifa.head()


# In[6]:


corr = fifa.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[7]:


fifa.isnull().sum()


# In[8]:


fifa.dtypes


# In[9]:



fifa_na_ = fifa.dropna()

X = fifa_na_.drop(columns='Overall').values
X_ = fifa_na_.drop(columns='Overall')
Y = fifa_na_.Overall
x = StandardScaler().fit_transform(X)
fifa_na = pd.DataFrame(x)
fifa_na.head()
print(type(x),type(Y))


# In[ ]:





# In[10]:


# data = fifa_na
# pca_095 = PCA(n_components=0.95)
# X_reduced = pca_095.fit_transform(data)

# X_reduced.shape # Segundo elemento da tupla é o número de componentes encontrados.


# In[11]:


data = fifa_na
# define transform
pca = PCA()
# prepare transform on dataset
pca.fit(fifa_na_)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.

component_number


# In[12]:


evr = pca.explained_variance_ratio_
evr


# In[13]:


data = fifa_na
# define transform
pca = PCA(n_components=2)
# prepare transform on dataset
pca.fit(data,Y)
# apply transform to dataset
df_ = pca.transform(data)
df = pd.DataFrame(pca.transform(data))
print(f"Original shape: {data.shape}, projected shape: {df.shape}")


# In[14]:


df.head()


# In[15]:


# evr = pca.explained_variance_ratio_
# evr


# In[16]:


# df.columns = df[:]
# df.head()


# In[17]:


g = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
g.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# In[18]:



# # Specify the features of interest and the target

# classes = ['Age', 'Potential', 'HeadingAccuracy', 'ShortPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Reactions', 'Stamina', 'Strength', 'Positioning', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking', 'GKReflexes']

# visualizer = PCA(scale=True,projection=3, classes=classes)
# visualizer.fit_transform(x, Y)

# visualizer.show()


# In[19]:


# # Load the concrete dataset

# visualizer = PCA(scale=True, proj_features=False,cmap='reset')
# visualizer.fit_transform(x, Y)
# visualizer.show()


# In[20]:


# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(X_)

# # Create the visualizer and draw the vectors
# tsne = TSNEVisualizer(decompose_by=2, cmap='reset')
# print(X.shape, Y.shape, X_.shape)
# tsne.fit(X, Y)
# tsne.show()




# In[21]:


fea = X_.loc[:,['Age', 'Potential', 'HeadingAccuracy', 'ShortPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Reactions', 'Stamina', 'Strength', 'Positioning', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking']]


# In[22]:


scaler = StandardScaler()
scaler.fit(fea)
X=scaler.transform(fea)    
pca = PCA()
x_new = pca.fit_transform(X)


# In[23]:


# def myplot(score,coeff,labels=None):
#     fig = plt.figure(figsize = (15,8))
#     cmap=plt.cm.get_cmap('hsv', 10)
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley, c = Y, cmap=cmap)
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'r', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
#     #plt.xlim(-1,1)
#     #plt.ylim(-1,1)
#     plt.xlabel("PC{}".format(1))
#     plt.ylabel("PC{}".format(2))
#     plt.grid()
#     jet = cm = plt.get_cmap('jet') 
# #Call the function. Use only the 2 PCs.
# myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))

# plt.colorbar()
# plt.show()


# In[ ]:





# In[24]:


X = fifa_na_.drop(columns='Overall')
lr = LinearRegression(normalize=True)
rfe = RFE(lr, n_features_to_select=5, verbose =3 ).fit(X,Y)
#ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
print(list(X.columns[rfe.support_]))


# In[25]:


# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("")
# plt.plot(range(1, len(rfe.support_) + 1), rfe.support_)
# plt.show()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[26]:


def q1():
    return float(round(evr[0],3))
    


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[27]:


def q2():
    return int(component_number)


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[28]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[29]:


def q3():
    pca = PCA(n_components=2).fit(fifa.dropna())
    pc = tuple(pca.components_.dot(x).round(3))
    return pc


# In[ ]:





# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[30]:


def q4():
    return list(X.columns[rfe.support_])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




