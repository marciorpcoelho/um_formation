import pandas as pd
import numpy as np
import os
import glob
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import FactorAnalysis
from gap_statistic import OptimalK
from xlsxwriter.workbook import Workbook


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


def save_csv(df, name):
    if os.path.isfile(name):
        os.remove(name)
    df.to_csv(name)


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


def csv_to_excel_converter(file):
    for csvfile in glob.glob(os.path.join('.', file)):
        workbook = Workbook(csvfile[:-4] + '.xlsx')
        worksheet = workbook.add_worksheet()
        with open(csvfile, 'rt', encoding='utf8') as f:
            reader = csv.reader(f)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet.write(r, c, col)
        workbook.close()


def unique_chassis_comparison(pse_sales, cm_bmw_mini):
    unique_chassis = pse_sales[pse_sales['nlr_code'] == '701']['chassis_number'].unique()
    print('Number of unique BMW/Mini Cars on PSE_Sales:', pse_sales[pse_sales['nlr_code'] == '701']['chassis_number'].nunique())
    print('Number of common chassis_numbers between PSE_Sales and CM BMW/MINI:', cm_bmw_mini[cm_bmw_mini['chassis_number'].isin(unique_chassis)].shape[0])