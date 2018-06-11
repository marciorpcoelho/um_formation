import pandas as pd
import time
import os
import numpy as np
import sys
from db_analysis import null_analysis, save_csv
pd.set_option('display.expand_frame_repr', False)
pd.options.mode.chained_assignment = None  # default='warn'


def vhe_sales_merge(*args):
    print('Merging VHE Sales across CA and CRP from DW and Current...')
    output_file = 'sql_db/' + 'BI_VHE_Sales.csv'

    dfs = []
    for file in args:
        df = pd.read_csv(file, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
        dfs.append(df)

    df = pd.concat(dfs)

    df.dropna(inplace=True)  # Since i don't know yet which columns are import for future matching, i'll just remove all NaN's. If some of the columns with NaN's are not relevant, then it is better to remove them, in order have more available data
    df.columns = map(str.lower, df.columns)
    df['registration_number'] = df['registration_number'].str.replace('-', '')
    df.drop(['vhe_code', 'vhe_number'], axis=1, inplace=True)

    if os.path.isfile(output_file):
        os.remove(output_file)
    df.to_csv(output_file)


def pse_sales_merge(*args):
    print('Merging PSE Sales across CA and CRP from DW and Current...')
    output_file = 'sql_db/' + 'BI_PSE_Sales.csv'

    dfs = []
    for file in args:
        df = pd.read_csv(file, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True, dtype={'nlr_code': object, 'Registration_Number': str, 'SLR_Document_Account': str, 'VAT_Number': object, 'PT_Franchise_Desc': str, 'PT_Model_Desc': str, 'PT_Sales_Type_Service_Level_1_Desc': str, 'PT_Sales_Type_Service_Level_2_Desc': str, 'slr_account': str, 'Chassis_Number': str})
        dfs.append(df)

    df = pd.concat(dfs)
    df.drop(['Anos_Viatura'], axis=1, inplace=True)  # I don't think i'll need this column, as i can just calculate it myself and it has over 27% missing info.
    df.dropna(inplace=True)  # Since i don't know yet which columns are import for future matching, i'll just remove all NaN's. If some of the columns with NaN's are not relevant, then it is better to remove them, in order have more available data
    df.columns = map(str.lower, df.columns)
    df['registration_number'] = df['registration_number'].str.replace('-', '')

    df = chassis_strip(df)

    if os.path.isfile(output_file):
        os.remove(output_file)
    df = pse_sales_cleanup(df)
    df.to_csv(output_file)


def chassis_strip(df):

    rows = []
    for row in df['chassis_number']:
        try:
            rows.append(row[-7:])
        except TypeError:
            rows.append(row)

    df['chassis_number'] = rows

    return df


def pse_sales_cleanup(df):
    df = df.rename(columns={'(no column name)': 'total'})
    df = df[df['registration_number'].apply(lambda x: len(x) == 6)]
    df = df[df['slr_account'] != '6']
    df = df[df['slr_account'] != '4']
    df = df[df['slr_account'] != '0']
    df = df[df['slr_account'] != '1']
    df = df[df['total'] > 0]

    return df


def cm_ca_cleanup(file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['DATA_INICIO', 'DATA_FIM', 'DATA_REMOCAO'], infer_datetime_format=True, usecols=['MATRICULA', 'ANOS', 'KILOMETROS', 'DATA_INICIO', 'DATA_FIM', 'DATA_REMOCAO', 'VIA_MAR_DES'])
    df = df.rename(columns={'MATRICULA': 'registration_number', 'ANOS': 'cm_years', 'KILOMETROS': 'cm_km', 'DATA_INICIO': 'cm_date_start', 'DATA_FIM': 'cm_date_end', 'DATA_REMOCAO': 'cm_date_removal', 'VIA_MAR_DES': 'pt_franchise_desc'})

    # df = df[df['registration_number'].apply(lambda x: len(x) == 6)]

    # Replacing end_date by end_removal when the second exists:
    df.update(pd.DataFrame({'cm_date_end': df['cm_date_removal']}))
    df.drop(['cm_date_removal'], axis=1, inplace=True)
    df.drop(['pt_franchise_desc'], axis=1, inplace=True)

    return df


def df_cleanup(df, pse=0, vhe=0):

    df.columns = map(str.lower, df.columns)
    df.dropna(subset=['registration_number'], axis=0, inplace=True)
    df.dropna(subset=['slr_account'], axis=0, inplace=True)
    df.dropna(subset=['registration_number', 'slr_account'], axis=0, inplace=True)
    df['registration_number'] = df['registration_number'].str.replace('-', '')
    df.drop(df[df['slr_account'] == 4].index, axis=0, inplace=True)
    df.drop(df[df['slr_account'] == 6].index, axis=0, inplace=True)
    # df.drop(df[df['slr_account'] == 0].index, axis=0, inplace=True)
    # df.drop(df[df['slr_account'] == 1].index, axis=0, inplace=True)
    df.drop(df[df['registration_number'] == 0].index, axis=0, inplace=True)
    df.drop(df[df['registration_number'] == '++++++'].index, axis=0, inplace=True)
    df.drop(df[df['registration_number'] == '000001'].index, axis=0, inplace=True)
    df = df[df['registration_number'].apply(lambda x: len(x) == 6)]
    df.rename(columns={'(no column name)': 'total'}, inplace=True)
    if pse:
        df = df[df['total'] > 0]
        df.drop('chassis_number', axis=1, inplace=True)
        df.drop('pt_model_desc', axis=1, inplace=True)
    if vhe:
        df.rename(columns={'slr_document_date': 'vhe_sold_date'}, inplace=True)
        df = chassis_strip(df)

    return df


def ca_db_merge(file11, file13):
    print('Merging CA DBs...')

    # df_ca = pd.read_csv(file11, delimiter=';', encoding='utf-8', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
    # df_dw_ca = pd.read_csv(file13, delimiter=';', encoding='utf-8', dtype={'Chassis_Number': str, 'Registration_Number': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
    df_ca = pd.read_csv(file11, delimiter=';', encoding='utf-8', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date_PSE', 'Registration_Date_VHE'], infer_datetime_format=True)
    df_dw_ca = pd.read_csv(file13, delimiter=';', encoding='utf-8', dtype={'Chassis_Number': str, 'Registration_Number': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date_PSE', 'Registration_Date_VHE'], infer_datetime_format=True)

    dfs = [df_ca, df_dw_ca]
    for df in dfs:
        df.columns = map(str.lower, df.columns)
        df['registration_number'] = df['registration_number'].str.replace('-', '')
        df.drop(df[df['slr_document_account'] == 4].index, axis=0, inplace=True)
        df.drop(df[df['slr_document_account'] == 6].index, axis=0, inplace=True)
        df.rename(columns={'(no column name)': 'total'}, inplace=True)
        df = chassis_strip(df)
        df = registration_number_flag(df)

    df_ca_concat = pd.concat(dfs, ignore_index=True)

    df_ca_grouped = df_ca_concat.groupby(['registration_number'])
    # df_ca['customer'] = df_ca_grouped['customer'].transform(lambda x: 'No values to aggregate' if pd.isnull(x).all() == True else x.fillna(method='ffill').fillna(method='bfill'))
    df_ca['customer'] = df_ca_grouped['customer'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df_ca['registration_date_pse'] = df_ca_grouped['registration_date_pse'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df_ca['registration_date_vhe'] = df_ca_grouped['registration_date_vhe'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # output_file = 'sql_db/' + 'ca.csv'
    save_csv(df_ca, 'sql_db/' + 'ca.csv')
    # if os.path.isfile(output_file):
    #     os.remove(output_file)
    # df_ca.to_csv(output_file)


def ca_cm_merge(file10):
    print('Adding CA CM DBs...')

    df = pd.read_csv('sql_db/' + 'ca.csv', index_col=0, parse_dates=['vehicle_in_date', 'registration_date_pse', 'registration_date_vhe'], infer_datetime_format=True)
    df_cm = cm_ca_cleanup(file10)

    df_merged = pd.merge(df, df_cm, how='left', on=['registration_number'], suffixes=('', '_y'))
    repeated_cols_left = [x for x in list(df_merged) if '_y' in x and x != 'cm_years']
    df_merged.drop(repeated_cols_left, axis=1, inplace=True)

    save_csv(df_merged, 'sql_db/' + 'ca_merged.csv')


def crp_db_merge(file12, file14):
    print('Merging CRP DBs...')

    # df_crp = pd.read_csv(file12, delimiter=';', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
    df_crp = pd.read_csv(file12, delimiter=';', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date_PSE', 'Registration_Date_VHE'], infer_datetime_format=True)
    # df_dw_crp = pd.read_csv(file14, delimiter=';', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
    df_dw_crp = pd.read_csv(file14, delimiter=';', dtype={'Chassis_Number': str, 'Registration_Number': str, 'Customer': str}, parse_dates=['Vehicle_In_Date', 'Registration_Date_PSE', 'Registration_Date_VHE'], infer_datetime_format=True)

    dfs = [df_crp, df_dw_crp]
    for df in dfs:
        df.columns = map(str.lower, df.columns)
        df['registration_number'] = df['registration_number'].str.replace('-', '')
        df.drop(df[df['slr_document_account'] == 4].index, axis=0, inplace=True)
        df.drop(df[df['slr_document_account'] == 6].index, axis=0, inplace=True)
        df.rename(columns={'(no column name)': 'total'}, inplace=True)
        df = chassis_strip(df)
        df = registration_number_flag(df)

    df_crp_concat = pd.concat(dfs, ignore_index=True)

    df_crp_grouped = df_crp_concat.groupby('registration_number')
    # df_crp['customer'] = df_crp_grouped['customer'].transform(lambda x: 'No values to aggregate' if pd.isnull(x).all() == True else x.fillna(method='ffill').fillna(method='bfill'))
    df_crp['customer'] = df_crp_grouped['customer'].apply(lambda x: x.ffill().bfill())
    df_crp['registration_date_pse'] = df_crp_grouped['registration_date_pse'].apply(lambda x: x.ffill().bfill())
    df_crp['registration_date_vhe'] = df_crp_grouped['registration_date_vhe'].apply(lambda x: x.ffill().bfill())


    # output_file = 'sql_db/' + 'crp.csv'
    # if os.path.isfile(output_file):
    #     os.remove(output_file)
    # df_crp.to_csv(output_file)
    save_csv(df_crp, 'sql_db/' + 'crp.csv')


def crp_cm_merge(file9):
    print('Adding CRP CM DBs...')
    df = pd.read_csv('sql_db/' + 'crp.csv', index_col=0, parse_dates=['vehicle_in_date', 'registration_date_pse', 'registration_date_vhe'], infer_datetime_format=True)
    df_cm = cm_crp_cleanup(file9)

    df_merged = pd.merge(df, df_cm, how='left', on=['chassis_number'], suffixes=('', '_y'))
    repeated_cols_left = [x for x in list(df_merged) if '_y' in x if x != 'cm_years']
    df_merged.drop(repeated_cols_left, axis=1, inplace=True)

    save_csv(df_merged, 'sql_db/' + 'crp_merged.csv')
    # df_merged.to_csv('sql_db/' + 'crp_merged.csv')


def registration_number_flag(df):

    df['registration_number_error_flag'] = 0
    df.dropna(subset=['registration_number'], axis=0, inplace=True)
    df_error = df[df['registration_number'].apply(lambda x: len(x) != 6)]
    df_error = df_error.append(df[df['registration_number'] == '000000'])
    df_error = df_error.append(df[df['registration_number'] == '000001'])
    df_error = df_error.append(df[df['registration_number'] == '++++++'])

    df.loc[df_error.index, 'registration_number_error_flag'] = 1

    return df


def cm_crp_cleanup(file):
    df = pd.read_csv(file, delimiter=',', encoding='latin-1', header=None, parse_dates=[4, 5], infer_datetime_format=True, dayfirst=True)
    df = df.rename(columns={1: 'chassis_number', 2: 'cm_months', 3: 'cm_km', 4: 'cm_date_start', 5: 'cm_date_end'})
    df['cm_years'] = df['cm_months'] * 1. / 12
    df.drop([0, 'cm_months'], axis=1, inplace=True)
    # df = pd.read_csv('sql_db/' + 'cm_bmw_mini_alt.csv', index_col=0, usecols=['Unnamed: 0', 'chassis_number', 'cm_date_start', 'cm_date_end'], parse_dates = ['cm_date_start', 'cm_date_end'], dayfirst = True, infer_datetime_format = True)
    # print(null_analysis(df))
    # print(df.head())

    return df


def db_concat():
    print('Concatenating CA and CRP DBs...')

    dtypes = {'nlr_code': int, 'slr_account': str, 'kms': int}
    parse_dates = ['vehicle_in_date', 'registration_date_pse', 'registration_date_vhe', 'cm_date_start', 'cm_date_end']

    db_ca = pd.read_csv('sql_db/' + 'ca_merged.csv', index_col=0, dtype=dtypes, parse_dates=parse_dates, infer_datetime_format=True)
    db_crp = pd.read_csv('sql_db/' + 'crp_merged.csv', index_col=0, dtype=dtypes, parse_dates=parse_dates, infer_datetime_format=True)

    # Filling for non-available data of CRP db
    # db_crp['cm_years'], db_crp['cm_km'] = 0, 0

    db = pd.concat([db_ca, db_crp], ignore_index=True, sort=False)
    first_cols = ['customer', 'slr_document_account', 'registration_number']
    rem_cols = [x for x in list(db) if x not in first_cols]
    cols = first_cols + rem_cols
    db = db[cols].sort_values(first_cols)

    save_csv(db, 'sql_db/' + 'db.csv')


def main():
    start = time.time()
    file1 = 'sql_db/' + 'BI_CA_VHE_Sales.csv'
    file2 = 'sql_db/' + 'BI_CRP_VHE_Sales.csv'
    file3 = 'sql_db/' + 'BI_CA_PSE_Sales.csv'
    file4 = 'sql_db/' + 'BI_CRP_PSE_Sales.csv'
    file5 = 'sql_db/' + 'BI_DW_CA_VHE_Sales.csv'
    file6 = 'sql_db/' + 'BI_DW_CRP_VHE_Sales.csv'
    file7 = 'sql_db/' + 'BI_DW_CA_PSE_Sales.csv'
    file8 = 'sql_db/' + 'BI_DW_CRP_PSE_Sales.csv'
    file9 = 'sql_db/' + 'BSI_2018053000.txt'  # Contratos de Manutenção BMW/Mini
    file10 = 'sql_db/' + 'contratos_manutencao_toyota_lexus.csv'
    # file11 = 'sql_db/' + 'ca_pse.csv'
    file11 = 'sql_db/' + 'ca_vhe_dates.csv'
    # file12 = 'sql_db/' + 'crp_pse.csv'
    file12 = 'sql_db/' + 'crp_vhe_dates.csv'
    # file13 = 'sql_db/' + 'dw_ca_pse.csv'
    file13 = 'sql_db/' + 'dw_ca_vhe_dates.csv'
    # file14 = 'sql_db/' + 'dw_crp_pse.csv'
    file14 = 'sql_db/' + 'dw_crp_vhe_dates.csv'

    vhe_sales = 0
    pse_sales = 0
    cm_toyota_lexus_cleanup = 0
    ca_merge = 0
    crp_merge = 0

    if vhe_sales:
        vhe_sales_merge(file1, file2, file5, file6)
    if pse_sales:
        pse_sales_merge(file3, file4, file7, file8)
    if cm_toyota_lexus_cleanup:
        cm_ca_cleanup(file10)
    if ca_merge:
        ca_db_merge(file11, file13)
        ca_cm_merge(file10)
    if crp_merge:
        crp_db_merge(file12, file14)
        crp_cm_merge(file9)

    if not ca_merge and not crp_merge:
        db_concat()
    print('\n', time.time() - start)


if __name__ == '__main__':
    main()
