##Stock Optimization


###Feature Treatment:  
Column - **Prov**:  

Before:  
![](./output/prov_before.png)

Grouping of values 'Viaturas Km 0' and 'Em Utilização':

After:  
![](./output/prov_after.png)


Column - **Cor_Exterior**: 
Before:  
![](./output/cor_exterior_before.png)

Grouping of values not in 'preto', 'cinzento', 'branco' and 'azul':
 
After:  
![](./output/cor_exterior_after.png)


Column - **Cor_Interior**:   
Before:  
![](./output/cor_interior_before.png)

Grouping of values not in 'preto', 'antracite', 'dakota' and 'antracite/cinza/preto':
 
After:  
![](./output/cor_interior_after.png)


Column - **Tipo Encomenda**:  
Before:  
![](./output/tipo_encomenda_before.png)

Grouping of values 'Demonstração', 'Serviço', 'Encom. Urgente':
 
After:  
![](./output/tipo_encomenda_after.png)


Column - **Jantes**:  
Before:  
![](./output/jantes_before.png)

Grouping of values '15', '16' and '20':
 
After:  
![](./output/jantes_after.png)

Column - **Local da Venda**  
Before:  
![](./output/local_da_venda_before.png)    

Grouping according to their location, new and used and vehicles/motorcycles:  

After:  
![](./output/local_da_venda_after.png)


Column - **Modelo**  
Before:  
![](./output/modelo_before.png)  

Grouping according to vehicle model:  

After:  
![](./output/modelo_after.png)


[//]: <> (### Feature Removal?)

[//]: <> (Columns 'Prov' and 'Tipo Encomenda' typically have higher scores, lower stock_days and higher margins. This translates to a higher correlation to our target variables, and a higher feature importance) 
[//]: <> (for these columns. Removing them decreases considerably the performance. Should we remove them?)
  
[//]: <> (E.g. of feature importance with and without 'Prov' and 'Tipo Encomenda' columns:)

[//]: <> (With:)  
[//]: <> (![](./output/feature_importance_target_score_class_group_cols.png)

[//]: <> (Without:)  
[//]: <> (![](./output/feature_importance_target_score_class_group_cols_prov_and_type.png)


###Model Performance
After grouping the classes and cleaning the dataset, three approaches were attempted: Classification (Decision Trees), Clustering (MiniBatchKMeans) and Classification (LinearRegression).  
Classification is evaluated in terms of Precision, Recall, F1 Score and Accuracy, for the different target variables.
Clustering is evaluated in terms of the silhouette value for each cluster, as well as cluster's size.
Regression is evaluated in terms of mean squared error and the coefficient R^2.
Explanations of each metric is followed.
  
**Evaluation Metrics:**  
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

[//]: <> (**Conditions:**  
**Standard** - Without grouped columns and without 'Prov' and 'Tipo Encomenda';  
**Grouping Columns** - With grouped columns and without 'Prov' and 'Tipo Encomenda';  
**w/Prov and Tipo Enc** - Without grouped columns and with 'Prov' and 'Tipo Encomenda';  
**Grouping Cols and w/Prov and Tipo Enc** - With grouped columns and with 'Prov' and 'Tipo Encomenda'.)  

**Targets:**  
**stock_class1** - converts 'stock_days' to classes, using business limits for each class;  
**stock_class2** - converts 'stock_days' to classes, using quartile limits for each class;  
**margem_class1** - converts 'margem_percentagem' to classes, using quartile limits for each class;  
**score** - normalizes both 'stock_days' and 'margem_percentagem' to scores between [0, 1] and multiplies between them.  
**score_class** - same as previous one, but the values are separated between 4 even classes. 


####Classification

With Oversample:
![](./output/classification_performance_oversampled.png)
  
Without Oversample:
![](./output/classification_performance.png)  

The results seem to show a higher performance of models when no oversample is applied. This is due to the models with oversample having average results in both classes (~0.6 recall)
while models without oversample have a very low results on the minority class and high results on the majority class.  
Focusing on non-oversampled models, Gradient models achieve the best results overall with over 0.85 recall (1.00 on class 1 and 0.00 on class 0), an f1-score of 0.81 (0.93 on class 1 and 0.00 on class 0).
These results are artificially high as what the model does is to predict all instances as class 1. Virtually useless.




[//]: <> (####Clustering)

[//]: <> (![](./output/stock_optimization_clustering_10_cluster.png)


[//]: <> (####Regression
**Score:**
R^2 = -0.403
MSE = 0.002)

[//]: <> (**Stock Days:**  
R^2 = -66.272  
MSE = 26078.370)

[//]: <> (**Margem (%:**)
[//]: <> (R^2 = -0.012)  
[//]: <> (MSE = 1.774)