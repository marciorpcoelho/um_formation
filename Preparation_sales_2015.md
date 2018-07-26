# Stock Optimization - Sales 2015

## Pre-Processement

### Feature Grouping:  

Column - **Prov**:  

Before:  
![](./output/prov_before.png)

Grouping of values 'Viaturas Km 0' and 'Em Utilização', drop of 'Demonstração' and 'Em Utilização':

After:  
![](./output/prov_after_mini_removal.png)


Column - **Cor_Exterior**: 
Before:  
![](./output/cor_exterior_before.png)

Grouping of values as follows:

```python
preto = ['preto']
cinzento = ['cinzento', 'prateado', 'prata', 'cinza']
branco = ['branco']
azul = ['azul', 'bluestone']
verde = ['verde']
vermelho_laranja = ['vermelho', 'laranja']
burgundy = ['burgundy']
castanho = ['castanho']
others = ['jatoba', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone']

```

After:  
![](./output/cor_exterior_after_mini_removal.png)


Column - **Cor_Interior**:   
Before:  
![](./output/cor_interior_before.png)

Grouping of values as follows:

```python
preto = ['preto', 'prata/preto/preto', 'veneto/preto', 'preto/preto', 'ambar/preto/preto']
antracite = ['antracite', 'antracite/cinza/preto', 'antracite/preto', 'antracite/vermelho/preto', 'antracite/vermelho', 'anthtacite/preto', 'anthracite/silver']
castanho = ['dakota', 'castanho', 'oak', 'terra', 'mokka', 'vernasca']
others = ['champagne', 'branco', 'oyster', 'prata/cinza', 'bege', 'oyster/preto', 'azul', 'cinzento', 'truffle', 'burgundy', 'zagora/preto', 'sonoma/preto', 'laranja', 'taupe/preto', 'vermelho', 'silverstone', 'nevada', 'cognac/preto', 'preto/laranja', 'preto/prateado']

```

After:  
![](./output/cor_interior_after_mini_removal.png)


Column - **Tipo Encomenda**  
Before:  
![](./output/tipo_encomenda_before.png)



After:  **Not used - Values are not trustworthy**.


Column - **Jantes**:  
Before:  
![](./output/jantes_before.png)

Grouping of values as follows:

```python
standard = ['standard', '15', '16']
seventeen_pol = ['17']
eighteen_pol = ['18']
nineteen_or_twenty_pol = ['19', '20']
```

After:  
![](./output/jantes_after_mini_removal.png)

Column - **Local da Venda**  
Before:  
![](./output/local_da_venda_before.png)    

Grouping according to their location as follows:

```python
centro = ['DCV - Coimbrões', 'DCC - Aveiro']
norte = ['DCC - Feira', 'DCG - Gaia', 'DCN-Porto', 'DCN-Porto Mini', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
sul = ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad', 'DCS-Lisboa Usados']
algarve = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados']
motorcycles = ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']

```

After:  
![](./output/local_da_venda_after_mini_removal.png)


Column - **Modelo**  
Before:  
![](./output/modelo_before.png)  

Grouping as follows:

```python
s2_gran = ['S2 Gran Tourer']
s2_active = ['S2 Active Tourer']
s3_touring = ['S3 Touring']
s3_berlina = ['S3 Berlina']
s4_gran = ['S4 Gran Coupé']
s5_touring = ['S5 Touring']
s5_lim_ber = ['S5 Limousine', 'S5 Berlina']
s1 = ['S1 3p', 'S1 5p']
x1 = ['X1']
x3 = ['X3 SUV']
mini_club = ['MINI CLUBMAN']
mini_cabrio = ['MINI CABRIO']
mini_country = ['MINI COUNTRYMAN']
mini = ['MINI 5p', 'MINI 3p']
motos = ['Série C', 'Série F', 'Série K', 'Série R']
outros = ['S2 Cabrio', 'S2 Gran Tourer', 'S2 Coupé', 'S3 Gran Turismo', 'S4 Coupé', 'S4 Cabrio', 'S5 Gran Turismo', 'S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé', 'S7 Berlina', 'S7 L Berlina', 'X2 SAC', 'X4 SUV', 'X5 SUV', 'X5 M', 'X6', 'X6 M', 'Z4 Roadster', 'M2 Coupé', 'M3 Berlina', 'M4 Cabrio', 'M4 Coupé', 'S6 Gran Turismo', 'S6 Cabrio', 'S6 Coupé', 'S6 Gran Coupe', 'S7 Berlina', 'S7 L Berlina']
```

After:  
![](./output/modelo_after_mini_removal.png)

### Data removal:

'Z4' model, motocycle and Mini entries were removed as the first is too recent while the others require different configurations. 

### Feature Engineering:

In this section, several new features were created in an attempt to give more insights to models to improve the results. These were the generated features and apply to each configuration:

**prev_sales_check:** boolean check for existence of previous sales;
**number_prev_sales**: the number of previous sales;
**last_score**: the last score obtained;
**last_margin**: the last margin obtained;
**last_stock_days**: the last number of stock days obtained;
**average_score_global**: the average of the scores obtained;
**min_score_global**: the minimum score ever obtained;
**max_score_global**: the maximum score ever obtained;
**q3_score_global**: the third quartile (0.75) of the distribution of the previous scores;
**median_score_global**: the second quartile (0.5) of the distribution of the previous scores;
**q1_score_global**: the first quartile (0.25) of the distribution of the previous scores;
**prev_score:** the last score obtained;
**average_score_dynamic:** the mean of all the previous scores;
**average_score_dynamic_std:** the standard deviation of the mean of all the previous scores;
**prev_average_score_dynamic:** the mean of all the previous scores before a specific sell is done;
**prev_average_score_dynamic_std:** the standard deviation of the mean of all the previous scores before a specific sell is done;

## Feature Selection

The reducion of features can be help performance by removing noisy or small variance features. 
In order to achieve this, multiple criteria were tested: **Chi Squared**, **ANOVA** or **Mutual Information**.



## Model Performance

After grouping the classes and cleaning the dataset, three approaches were attempted: Classification (Decision Trees), Clustering (MiniBatchKMeans) and Classification (LinearRegression).  
Classification is evaluated in terms of Precision, Recall, F1 Score and Accuracy, for the different target variables.
Clustering is evaluated in terms of the silhouette value for each cluster, as well as cluster's size.
Regression is evaluated in terms of mean squared error and the coefficient R^2.
Explanations of each metric is followed.

#### **Evaluation Metrics:**  

**Accuracy, [0,1]** - Of all the classified, how many are correct?;
**Precision, [0,1]** - Of all the classified as a class, how many are correct?;  
**Recall/Sensitivity/TPR, [0,1]** - Of the whole positive class, how many were correctly predicted?;
**Specificity/TNR, [0, 1]**:  Of the whole negative class, how many were correctly predicted?
**False Positive Rate, [0, 1]:** 1 - TNR
**F1-Score, [0,1]** - Harmonic mean of Precision and Recall;
**Micro F1-Score, [0, 1]:** Calculates the precision and recall across all classes counting all the true positives, false negatives and false positives;
**Macro F1-Score, [0, 1]:** Calculates the precision and recall for each class and calculates its mean;



#### **Targets:**  

**stock_class1** - converts 'stock_days' to classes, using business limits for each class;  
**stock_class2** - converts 'stock_days' to classes, using quartile limits for each class;  
**margem_class1** - converts 'margem_percentagem' to classes, using quartile limits for each class;  
**score_class** - same as previous one, but the values are separated between 4 even classes;
**score** - normalizes both 'stock_days' and 'margem_percentagem' to scores between [0, 1] and multiplies between them;
**new_score** - normalizes both 'stock_days' and 'margem_percentagem' to scores between [0, 1] and classifies each one in two classes 0 and 1. Currently, 'stock_days' with class 1 correspond to configurations with less than 30 days in stock, while class 1 in 'margem_percentagem' refers to configurations with a positive margin. After this discretization, the end target will have as class 1 only entries that were previously classified as 1 in both metrics, ensuring minimum values for each dimension;



## Classification

Comparison between with and without Oversample, before feature engineering (2017+ data only): 

Without Oversample:  
![](./output/Backup/classification_performance_recall.png)

With Oversample:  
![](./output/Backup/classification_performance_recall_oversampled.png)

The results seem to show a higher performance of models when no oversample is applied. This is due to the models with oversample having average results in both classes (~0.6 recall)
while models without oversample have a very low results on the minority class and high results on the majority class.  
Focusing on non-oversampled models, Gradient models achieve the best results overall with over 0.85 recall (1.00 on class 1 and 0.00 on class 0), an f1-score of 0.81 (0.93 on class 1 and 0.00 on class 0).
These results are artificially high as what the model does is to predict all instances as class 1. Virtually useless. 



After this initial approach, more data was made available, new target scores were created and new features were implemented (check above). With these changes, the results are as follows:

![](C:\Users\mrpc\PycharmProjects\um_formation\output\5_classification_performance_target_new_score_scoring_recall.png)

The results show that typically only meta-models (Adaboost, Random Forest and Voting) reach better scores  across all metrics (precision, average recall, F1 score and also recall for both classes). The ROC curve for all these models are:

![](C:\Users\mrpc\PycharmProjects\um_formation\output\roc_curve_classification_voting_target_new_score_scoring_recall.png)

Where again Adaboost, Voting and Random Forest achieve the best results. Looking at the normalized confusion matrices:

![](C:\Users\mrpc\PycharmProjects\um_formation\output\confusion_matrix_normalized_classification_voting_target_new_score_scoring_recall.png)