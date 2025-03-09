#!/usr/bin/env python
# coding: utf-8

# ## Importação de bibliotecas

# In[545]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


# ## Vizualizar dados e carregar

# In[546]:


df = pd.read_csv('Tabela_final.csv')
df.head()


# ## Pré-processamento

# In[547]:


df['data'] = pd.to_datetime(df['data'])
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year


# ## Features e target
# Features sao as variaveis independentes(mes) e o target é nossa varivel objetivo(temp, chuva e Icidencia de radiação solar)

# In[548]:


X = df[['mes']]
y_temp = df['Temperatura (°C)']
y_chuva = df['Chuva (mm)']
y_rad = df['Solar Radiation']
y_po = df['IQA']


# ## Divisão de Treino e teste 

# In[549]:


X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size = 0.2, random_state = 42)
X_train, X_test, y_chuva_train, y_chuva_test = train_test_split(X, y_chuva, test_size = 0.2, random_state = 42)
X_train, X_test, y_rad_train, y_rad_test = train_test_split(X, y_rad, test_size = 0.2, random_state = 42)
X_train, X_test, y_po_train, y_po_test = train_test_split(X, y_po, test_size = 0.2, random_state = 42)


# ## Treinando modelo SVR (Support Vector Regression)

# In[550]:


model_temp = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
model_temp.fit(X_train, y_temp_train)

model_chuva = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
model_chuva.fit(X_train, y_chuva_train)

model_rad = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
model_rad.fit(X_train, y_rad_train)

model_po = RandomForestRegressor(random_state=42)
model_po.fit(X_train, y_po_train)


# ## previsionamento para 2025

# In[551]:


meses_2025 = pd.DataFrame({'mes': range(1,13)})

prev_temp_2025 = model_temp.predict(meses_2025)
prev_chuva_2025 = model_chuva.predict(meses_2025)
prev_rad_2025 = model_rad.predict(meses_2025)
prev_po_2025 = model_po.predict(meses_2025)

meses_2025['Temperatura (°C)'] = prev_temp_2025
meses_2025['Chuva (mm)'] = prev_chuva_2025
meses_2025['Solar Radiation'] = prev_rad_2025
meses_2025['IQA'] = prev_po_2025


# ## Resultados

# In[554]:


meses_2024 = df[df['ano'] == 2024].groupby('mes').mean()
def calc_tendencia(series_2025, series_2024, nome):
    if series_2024.isnull().all() or series_2025.isnull().all():
        print(f"Os dados de {nome} contêm valores nulos ou não existem.")
        return
    
    if series_2024.empty or series_2025.empty:
        print(f"Não há dados suficientes de {nome} para comparação entre os anos.")
        return

    # Criando modelo de regressão para detectar tendência
    meses = np.array(range(1, 13)).reshape(-1, 1)
    modelo_2025 = LinearRegression().fit(meses, series_2025)
    coef_2025 = modelo_2025.coef_[0]  # Coeficiente angular da reta de tendência
    
    direcao = "aumentando" if coef_2025 > 0 else "diminuindo"


    # Comparação de faixas de valores
    print(f"Faixa de valores de {nome} em 2025: {series_2025.min():.2f} a {series_2025.max():.2f}")
    print(f"Faixa de valores de {nome} em 2024: {series_2024.min():.2f} a {series_2024.max():.2f}")

    # Comparação da média anual
    diferenca_media = series_2025.mean() - series_2024.mean()
    variacao = "maior" if diferenca_media > 0 else "menor"

    print(f"Em média, {nome} está {variacao} em 2025 do que em 2024 por {abs(diferenca_media):.2f} unidades.\n")

# Executar a análise para cada variável
calc_tendencia(meses_2025['Temperatura (°C)'], meses_2024['Temperatura (°C)'], "Temperatura")
calc_tendencia(meses_2025['Chuva (mm)'], meses_2024['Chuva (mm)'], "Chuva")
calc_tendencia(meses_2025['Solar Radiation'], meses_2024['Solar Radiation'], "Radiação Solar")
calc_tendencia(meses_2025['IQA'], meses_2024['IQA'], "Poluição")


# ## Vizualizar as previsões graficamente (Temperatura)

# In[539]:


plt.figure(figsize=(6, 5))
plt.plot(meses_2025['mes'], meses_2025['Temperatura (°C)'], marker='o', linestyle='-', color='red', label='Temperatura (°C)')
plt.title('Previsão de Temperatura Média para 2025', fontsize=16)
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# ## Vizualizar as previsões graficamente (Chuva mm)

# In[522]:


plt.figure(figsize=(6, 5))
plt.bar(meses_2025['mes'], meses_2025['Chuva (mm)'], color='blue', label='Chuva (mm)')
plt.title('Previsão de Chuva para 2025', fontsize=16)
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Chuva (mm)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# ## Avaliar graficamente a Incidencia Solar

# In[568]:


plt.figure(figsize=(6, 5))
plt.bar(meses_2025['mes'], meses_2025['Solar Radiation'], color='yellow', label='Incidência Solar')
plt.title('Previsão de Incidência Solar para 2025', fontsize=16)
plt.xlabel('Mês', fontsize=12)
plt.ylabel('índice de Incidência solar', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# ## Avaliar graficamente poluição

# In[562]:


plt.figure(figsize=(6, 6))  # Aumentar a largura da figura
bars = plt.barh(meses_2025.index, meses_2025['IQA'], color='green', label='IQA')

plt.title('Previsão da Poluição média para 2025', fontsize=16, pad=20)
plt.xlabel('IQA', fontsize=12)
plt.ylabel('Mês', fontsize=12)

meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
plt.yticks(meses_2025.index, meses_nomes, fontsize=12)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.0f}', 
             va='center', ha='left', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()


# ## Avaliar as previsoes graficamente (Climograma)

# In[563]:


fig, ax1 = plt.subplots(figsize=(6, 6))
ax1.bar(meses_2025['mes'], meses_2025['Chuva (mm)'], color='blue', label='Chuva (mm)') 
ax2 = ax1.twinx()  
ax2.plot(meses_2025['mes'], meses_2025['Temperatura (°C)'], marker='o', linestyle='-', color='red', label='Temperatura (°C)')
ax1.set_xlabel('Mês', fontsize=12)
ax1.set_ylabel('Chuva (mm)', fontsize=12, color='blue')
ax2.set_ylabel('Temperatura (°C)', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='black')
plt.title('Climograma: Temperatura e Chuva para 2025', fontsize=16)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True, linestyle='--', alpha=0.7)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2
ax1.legend(lines, labels, loc='upper right')
plt.show()


# ## Avaliar o Modelo criado

# In[564]:


mae_temp = mean_absolute_error(y_temp_test, y_temp_pred)
rmse_temp = np.sqrt(mean_squared_error(y_temp_test, y_temp_pred))
mape_temp = np.mean(np.abs((y_temp_test - y_temp_pred) / y_temp_test)) * 100
mae_chuva = mean_absolute_error(y_chuva_test, y_chuva_pred)
rmse_chuva = np.sqrt(mean_squared_error(y_chuva_test, y_chuva_pred))
mape_chuva = np.mean(np.abs((y_chuva_test - y_chuva_pred) / y_chuva_test)) * 100
mae_rad = mean_absolute_error(y_rad_test, y_rad_pred)
rmse_rad = np.sqrt(mean_squared_error(y_rad_test, y_rad_pred))
mape_rad = np.mean(np.abs((y_rad_test - y_rad_pred) / y_rad_test)) * 100
mae_po = mean_absolute_error(y_po_test, y_po_pred)
rmse_po = np.sqrt(mean_squared_error(y_po_test, y_po_pred))
mape_po = np.mean(np.abs((y_po_test - y_po_pred) / y_po_test)) * 100

errors_df = pd.DataFrame({
    'Temperatura': np.abs(y_temp_test - y_temp_pred),
    'Chuva': np.abs(y_chuva_test - y_chuva_pred),
    'Radiação': np.abs(y_rad_test - y_rad_pred),
    'Poluição': np.abs(y_po_test - y_po_pred)
})

plt.figure(figsize=(6, 6))
plt.boxplot(errors_df.values, tick_labels=errors_df.columns)
plt.title('Distribuição dos Erros de Predição')
plt.ylabel('Erro Absoluto')
plt.savefig('prev_error.png', dpi = 400)
plt.show()

print("Erro Médio Absoluto (MAE):")
print(f"Temperatura: {mae_temp:.2f}")
print(f"Chuva: {mae_chuva:.2f}")
print(f"Radiação: {mae_rad:.2f}")
print(f"Poluição: {mae_po:.2f}")
print("\n")

print("Raiz do Erro Quadrático Médio (RMSE):")
print(f"Temperatura: {rmse_temp:.2f}")
print(f"Chuva: {rmse_chuva:.2f}")
print(f"Radiação: {rmse_rad:.2f}")
print(f"Poluição: {rmse_po:.2f}")
print("\n")

print("Erro Percentual Médio Absoluto (MAPE):")
print(f"Temperatura: {mape_temp:.2f}%")
print(f"Chuva: {mape_chuva:.2f}%")
print(f"Radiação: {mape_rad:.2f}%")
print(f"Poluição: {mape_po:.2f}%")


# In[ ]:





# In[ ]:





# In[ ]:




