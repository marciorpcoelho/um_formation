import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import FactorAnalysis
from gap_statistic import OptimalK


def null_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: '#null:'}))
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.rename(index={0: '%null:'}))

    print(tab_info)


def zero_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0)).T.rename(index={0: '#null:'}))
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0) / df.shape[0] * 100).T.rename(index={0: '%null:'}))

    print(tab_info)


def const_col_removal(df):

    list_before = list(df)
    for column in list_before:
        if (df[column].nunique() == 1):
            # print(column)
        # if (df[df[column] == 0][column].shape[0] * 1.) / df.shape[0] == 1:
        #     print(column)
            df.drop(column, axis=1, inplace=True)
    list_after = list(df)
    print('Constant-columns removal:', [x for x in list_before if x not in list_after], '\n')

    return df


def df_standardization(df):

    scaler = StandardScaler()
    scaler.fit(df)
    scaled_matrix = scaler.transform(df)

    return scaled_matrix


def feature_selection(df, number_features):
    print('Feature Selection')

    selector = SelectKBest(mutual_info_regression, k=number_features).fit()


def gap_optimalk(matrix):

    optimalk = OptimalK(parallel_backend='joblib')
    k = optimalk(matrix, cluster_array=np.arange(1, 20))
    print('\nOptimal number of clusters is ', k)

    return k