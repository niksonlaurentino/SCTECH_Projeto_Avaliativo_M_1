**Risco de Crédito — Previsão de Inadimplência para setor bancário**

📘 **Projeto Avaliativo do SENAI - Módulo 1**

🤖 **Machine Learning que prevê se um cliente de um banco se tornará inadimplente (loan_status = 1) ou pagará o empréstimo em dia (loan_status = 0), comparando dois algoritmos de classificação: KNN e Random Forest.**
    
📈 **Desafio de Negócio**
Qual o impacto financeiro para o banco se o modelo classificar um bom pagador como "Risco de Calote" (Falso Positivo) ou um mau pagador como "Seguro" (Falso Negativo)?


💸 **Erro de predição significa perda financeira substancial.**

Falso Negativo (FN)	Crédito liberado para inadimplentes futuros — erro mais crítico (além de não receber, deixa de emprestar o mesmo capital)
Falso Positivo (FP)	Crédito negado a potenciais pagadores -	Custo de oportunidade (spread não capturado)

📂 **Dataset**

•	Arquivo: credit_risk_dataset.csv
•	Tamanho: 32.581 linhas × 12 colunas (8 numéricas e 4 categóricas)
•	Target de predição: loan_status (0 = adimplente, 1 = inadimplente), com desbalanceamento de classes (utilizado o SMOTE para balanceamento sintético)

**1. Análise Exploratória (EDA)**
•	Dimensões, tipos de dados, describe() e frequência de valores
•	Histograma de idades, gráfico do desbalanceamento do alvo
•	Matrizes de correlação de Pearson e Spearman
•	Boxplots em escala logarítmica para evidenciar outliers

🔍 **Principais achados:**
•	Idades inválidas (3 registros de 144 anos e 2 de 123 anos) → erro de cadastro
•	Forte disparidade entre média e mediana de person_income → muitos outliers
•	Nulos em person_emp_length (2,74%) e loan_int_rate (9,55%)
•	Alta correlação entre person_age e cb_person_cred_hist_length (risco de multicolinearidade para o KNN)

**2. Limpeza e Engenharia de Atributos**
•	Remoção de duplicatas e de idades > 110 anos
•	Nova feature: comprometimento_renda = loan_amnt / person_income × 100
•	Bifurcação em dois fluxos de dados, um para cada algoritmo

**3. Pipelines**
Etapa	KNN	Random Forest
Colunas removidas	cb_person_cred_hist_length	—
Outliers	Removidos no treino (regra do IQR, 1,5×)	Mantidos
Imputação	Mediana (dentro do pipeline, sem data leakage)	Mediana
Encoding	OrdinalEncoder (loan_grade) + OneHotEncoder	OrdinalEncoder (loan_grade) + OneHotEncoder
Escalonamento StandardScaler, SMOTE (imblearn.pipeline) para KNN e class_weight='balanced' para RANDOM FOREST

**4. Otimização de Hiperparâmetros**
Estratégia em duas fases, com StratifiedKFold (5 dobras) e métrica F1:
  1.	RandomizedSearchCV — exploração ampla do espaço de busca
  2.	GridSearchCV — refinamento focado ao redor dos melhores valores

**5. Diagnóstico de Overfitting**
Auditoria com validação cruzada estratificada (5-fold), avaliando média e desvio padrão do F1 por dobra, com gráfico. 📊
<img width="984" height="484" alt="image" src="https://github.com/user-attachments/assets/383bc71a-cc87-46ec-932b-4b22649e9eba" />
<img width="984" height="484" alt="image" src="https://github.com/user-attachments/assets/7e04aebd-1e9c-44b9-bc32-3f7d2ba6b75a" />



📊 **Resultados**

<img width="1284" height="492" alt="image" src="https://github.com/user-attachments/assets/7ab8be47-336a-4dd6-aae6-ca9c71925eef" />


✅ **O Random Forest foi superior nos dois tipos de erro: reduziu os Falsos Negativos (o erro mais caro para o banco) e cometeu cerca de um terço dos Falsos Positivos do KNN. Esse foi o algoritmo escolhido.
**
🛠️ **Tecnologias**
•	Python 3.12
•	pandas, NumPy
•	matplotlib, seaborn
•	scikit-learn
•	imbalanced-learn


📝  **Observações**

•	O pré-processamento (imputação, encoding, SMOTE e escalonamento) fica dentro dos pipelines, evitando data leakage entre treino e validação.
•	O train_test_split usa stratify=y e random_state=42 para garantir reprodutibilidade e preservar a proporção das classes.
•	A remoção de outliers é aplicada somente ao treino do KNN; o conjunto de teste permanece intacto.
