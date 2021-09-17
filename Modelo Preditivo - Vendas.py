#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio

# #### Importar a Base de dados

# In[1]:


import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[2]:


import matplotlib.pyplot as plt # pacote de gráfico
import seaborn as sns  # pacote de gráfico

sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
plt.show()

sns.pairplot(tabela)
plt.show()


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[3]:


from sklearn.model_selection import train_test_split

# separar dados de X e de Y

# y - é quem a gente quer descobrir
y = tabela["Vendas"]

# x - é o resto
x = tabela.drop("Vendas", axis=1)

# aplicar o train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_randomforest = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_randomforest.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[5]:


previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_randomforest = modelo_randomforest.predict(x_teste)

from sklearn import metrics

# R2 > 0% a 100%, quanro maior, melhor
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_randomforest))


# #### Visualização Gráfica das Previsões

# In[7]:


# RandomForest é o melhor modelo
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["regressao linear"] = previsao_regressaolinear
tabela_auxiliar["random forest"] = previsao_randomforest

plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Qual a importância de cada variável para as vendas?

# In[ ]:


# como fazer uma nova previsao
# importar a nova_tabela com o produtos (a nova tabela tem que ter os dados de TV, Radio e Jornal)
modelo_randomforest.predict(nova_tabela)
print(previsao)


# In[10]:


sns.barplot(x=x_treino.columns, y=modelo_randomforest.feature_importances_)
plt.show()


# In[ ]:




