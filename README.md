Previsão de Renda (Adult Census Dataset)

Este é um projeto de Machine Learning para prever se a renda de um indivíduo é superior ou inferior a 50 mil dólares anuais, com base em dados demográficos do censo dos EUA.

O objetivo principal foi praticar o processo completo de um projeto de Data Science: desde a limpeza e preparação dos dados (Data Cleaning e Feature Engineering) até o treinamento e avaliação de um modelo de classificação.

🚀 O Processo

O notebook salary.ipynb segue uma jornada clara de análise e modelagem.

1. Limpeza e Preparação dos Dados (Passos 1 e 2)
O dataset original (adult11.csv) não estava pronto para o modelo:

Valores Ausentes: Os valores ausentes estavam rotulados como ' ?' (com espaços). Foi usada uma expressão regular (r'\s*\?\s*') para localizá-los, substituí-los por NaN e, em seguida, remover as linhas com dados faltantes.

Variável Alvo: A coluna alvo salary era um texto (' <=50K' e ' >50K'). Ela foi transformada em uma coluna numérica binária (salary_numeric), onde 0 representa <=50K e 1 representa >50K.

2. Engenharia de Features (Passo 3)
O modelo de Regressão Logística só aceita números, mas o dataset possuía 8 colunas de texto (categóricas), como workclass, occupation e marital-status.

One-Hot Encoding: Foi aplicada a técnica One-Hot Encoding (usando pandas.get_dummies) para converter essas colunas categóricas em múltiplas colunas numéricas (0 ou 1).

Prevenção de Multicolinearidade: O parâmetro drop_first=True foi usado para evitar a "Armadilha da Variável Dummy", garantindo a independência das features.

3. Modelagem e Treinamento (Passo 4)
Com os dados 100% numéricos, o modelo pôde ser treinado.

Escolha do Modelo: Como o problema é prever uma categoria (0 ou 1), o modelo ideal escolhido foi a Regressão Logística (LogisticRegression), que é mais adequado para classificação do que a Regressão Linear.

Padronização: As features foram padronizadas com StandardScaler. Isso é crucial para modelos como a Regressão Logística, pois coloca todas as features (como age e capital-gain) na mesma escala, melhorando a performance e a velocidade de treinamento.

Divisão: Os dados foram divididos em 80% para treino e 20% para teste (train_test_split).

📈 Resultados
O modelo final de Regressão Logística alcançou uma Acurácia (Accuracy) de 85.11% nos dados de teste.

O desempenho detalhado pode ser visto no Relatório de Classificação:

              precision    recall  f1-score   support

   <=50K (0)       0.88      0.93      0.90      6842
    >50K (1)       0.74      0.60      0.66      2203

    accuracy                           0.85      9045
   macro avg       0.81      0.77      0.78      9045
weighted avg       0.84      0.85      0.85      9045
Conclusões das Métricas:
Acurácia (85%): O modelo acerta a previsão de renda em 85% dos casos.

Precisão (Precision >50K = 0.74): De todas as vezes que o modelo previu que alguém ganhava >50K, ele estava correto 74% das vezes.

Recall (>50K = 0.60): O modelo conseguiu identificar corretamente 60% de todas as pessoas que realmente ganhavam >50K.
