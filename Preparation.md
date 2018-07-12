# Stock Optimization

## Feature Treatment:  

Column - **Prov**:  

Before:  
![](./output/prov_before.png)

Grouping of values 'Viaturas Km 0' and 'Em Utilização', drop of 'Demonstração' and 'Em Utilização':

After:  
![](./output/prov_after.png)


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
others = ['jatoba', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone']
```

After:  
![](./output/cor_exterior_after.png)


Column - **Cor_Interior**:   
Before:  
![](./output/cor_interior_before.png)

Grouping of values as follows:

```python
preto = ['preto', 'prata/preto/preto', 'veneto/preto', 'preto/preto', 'ambar/preto/preto']
antracite = ['antracite', 'antracite/cinza/preto', 'antracite/preto', 'antracite/vermelho/preto']
castanho = ['dakota', 'castanho', 'oak', 'terra', 'mokka']
others = ['branco', 'oyster', 'bege', 'oyster/preto', 'azul', 'cinzento', 'truffle', 'burgundy', 'zagora/preto', 'sonoma/preto', 'laranja', 'taupe/preto', 'vermelho', 'silverstone', 'nevada', 'cognac/preto', 'preto/laranja']
```

After:  
![](./output/cor_interior_after.png)


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
![](./output/jantes_after.png)

Column - **Local da Venda**  
Before:  
![](./output/local_da_venda_before.png)    

Grouping according to their location as follows:

```python
centro = ['DCV - Coimbrões', 'DCC - Aveiro']
norte = ['DCC - Feira', 'DCG - Gaia', 'DCN-Porto', 'DCN-Porto Mini', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
sul = ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad']
algarve = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados']
motorcycles = ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']
```

After:  
![](./output/local_da_venda_after.png)


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
![](./output/modelo_after.png)

Aditional preprocessing:

'Z4' model and motocycle entries were removed as the first is too recent while the latter requires different configurations. 

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

**Precision, [0,1]** - Of all the classified as a class, how many are correct?;  
**Recall, [0,1]** - Of the whole class, how many were correctly predicted?;  
**F1-Score, [0,1]** - Harmonic mean of Precision and Recall;  
**Accuracy, [0,1]** - Of all the classified, how many are correct?;  
**Silhouette Value, [-1,1]** - Gives a score to each sample, which measures how close that sample is to the boundary of neighbouring clusters. 
A value close to +1 indicates the sample is far away from the neighbouring clusters. A value of 0 indicates the sample is on or very close to the decision boundary between two neighbouring clusters,
and negative values indicate that those samples might have been assigned to the wrong cluster;  
**Cluster's Size [0, +inf]** - Indicates the number of samples per cluster. Clusters with too many or too few samples are to avoided, and the number of clusters should be changed;  
**Mean Squared Error [0, 1]** - Returns a measure of the average squared difference between the estimated value and what is estimated.  
**Coefficient R^2 [-inf, 1]** - Statistical measure of how close the data are to the fitted regression line. The best value is 1, while 0 means the model is constant and takes no input from the data.
A negative value means the models is arbitrarily worse than the null-hypothesis.

#### **Targets:**  

**stock_class1** - converts 'stock_days' to classes, using business limits for each class;  
**stock_class2** - converts 'stock_days' to classes, using quartile limits for each class;  
**margem_class1** - converts 'margem_percentagem' to classes, using quartile limits for each class;  
**score** - normalizes both 'stock_days' and 'margem_percentagem' to scores between [0, 1] and multiplies between them.  
**score_class** - same as previous one, but the values are separated between 4 even classes. 

## Classification

Comparison between with and without Oversample:  

Without Oversample:  
![](./output/Backup/classification_performance_recall.png)

With Oversample:  
![](./output/Backup/classification_performance_recall_oversampled.png)

The results seem to show a higher performance of models when no oversample is applied. This is due to the models with oversample having average results in both classes (~0.6 recall)
while models without oversample have a very low results on the minority class and high results on the majority class.  
Focusing on non-oversampled models, Gradient models achieve the best results overall with over 0.85 recall (1.00 on class 1 and 0.00 on class 0), an f1-score of 0.81 (0.93 on class 1 and 0.00 on class 0).
These results are artificially high as what the model does is to predict all instances as class 1. Virtually useless.  

With Mutual Information as Feature Selection Criteria:
![](./output/classification_performance_recall_oversampled_mutual_info_classif.png)

With F-Score as Feature Selection Criteria:
![](./output/classification_performance_recall_oversampled_f_classif.png)

With Chi Squared as Feature Selection Criteria:
![](./output/classification_performance_recall_oversampled_chi2.png)


