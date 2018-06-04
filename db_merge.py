import pandas as pd
import time
import os
import sys
from db_analysis import null_analysis
pd.set_option('display.expand_frame_repr', False)


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
        df = pd.read_csv(file, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True, dtype={'nlr_code': object, 'Registration_Number': str, 'SLR_Document_Account': str, 'VAT_Number': object, 'PT_Franchise_Desc': str, 'PT_Model_Desc': str, 'PT_Sales_Type_Service_Level_1_Desc': str, 'PT_Sales_Type_Service_Level_2_Desc': str})
        dfs.append(df)

    df = pd.concat(dfs)
    df.drop(['Anos_Viatura'], axis=1, inplace=True)  # I don't think i'll need this column, as i can just calculate it myself and it has over 27% missing info.
    df.dropna(inplace=True)  # Since i don't know yet which columns are import for future matching, i'll just remove all NaN's. If some of the columns with NaN's are not relevant, then it is better to remove them, in order have more available data
    df.columns = map(str.lower, df.columns)
    df['registration_number'] = df['registration_number'].str.replace('-', '')

    if os.path.isfile(output_file):
        os.remove(output_file)
    df = pse_sales_cleanup(df)
    df.to_csv(output_file)


def pse_sales_cleanup(df):
    df = df.rename(columns={'(no column name)': 'total'})
    df = df[df['registration_number'].apply(lambda x: len(x) == 6)]
    df = df[df['slr_account'] != '6']
    df = df[df['slr_account'] != '4']
    df = df[df['slr_account'] != '0']
    df = df[df['slr_account'] != '1']
    df = df[df['total'] > 0]

    return df


def cm_cleanup(file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['DATA_INICIO', 'DATA_FIM', 'DATA_REMOCAO'], infer_datetime_format=True, usecols=['MATRICULA', 'ANOS', 'KILOMETROS', 'DATA_INICIO', 'DATA_FIM', 'DATA_REMOCAO', 'VIA_MAR_DES'])
    df = df.rename(columns={'MATRICULA': 'registration_number', 'ANOS': 'cm_years', 'KILOMETROS': 'cm_km', 'DATA_INICIO': 'cm_date_start', 'DATA_FIM': 'cm_date_end', 'DATA_REMOCAO': 'cm_date_removal', 'VIA_MAR_DES': 'pt_franchise_desc'})

    df = df[df['registration_number'].apply(lambda x: len(x) == 6)]

    df.to_csv('sql_db/' + 'cm_toyota_lexus.csv')

    return df


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

    vhe_sales = 0
    pse_sales = 1
    cm_toyota_lexus_cleanup = 0

    if vhe_sales:
        vhe_sales_merge(file1, file2, file5, file6)
    if pse_sales:
        pse_sales_merge(file3, file4, file7, file8)
    if cm_toyota_lexus_cleanup:
        cm_cleanup(file10)

    print(time.time() - start)


if __name__ == '__main__':
    main()
