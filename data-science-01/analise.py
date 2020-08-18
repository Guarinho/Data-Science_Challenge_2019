import pandas as pd
import json

csv_path = open("desafio1.csv",'r')
df  = pd.read_csv(csv_path)

df_clean = df.drop(['id','sobrenome','genero','idade','nivel_estabilidade','saldo_conta','numero_produtos','possui_cartao_de_credito','membro_ativo','RowNumber'],axis=1)

N_estados = df_clean["estado_residencia"].unique()

a=df_clean.groupby(["estado_residencia"]).median().rename({'pontuacao_credito': "mediana"},axis=1).reset_index()
b=df_clean.groupby(["estado_residencia"]).mean().rename({'pontuacao_credito': "media"},axis=1).reset_index()
d=df_clean.groupby(["estado_residencia"]).std().rename({'pontuacao_credito': "desvio_padrao"},axis=1).reset_index()
c=df_clean.groupby(["estado_residencia"]).get_group('PR').mode().rename({'pontuacao_credito': "moda"},axis=1)
e=df_clean.groupby(["estado_residencia"]).get_group('SC').mode().rename({'pontuacao_credito': "moda"},axis=1)
f=df_clean.groupby(["estado_residencia"]).get_group('RS').mode().rename({'pontuacao_credito': "moda"},axis=1)


PR=c.merge(a).merge(b).merge(d).set_index("estado_residencia")
SC=e.merge(a).merge(b).merge(d).set_index("estado_residencia")
RS=f.merge(a).merge(b).merge(d).set_index("estado_residencia")

an = SC.append(RS).append(PR).to_dict('index')


import json
with open('submission.json', 'w', encoding='utf-8') as f:
    json.dump(an, f, ensure_ascii=False, indent=4)

