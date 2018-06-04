import pandas as pd
import time
pd.set_option('display.expand_frame_repr', False)
from db_analysis import null_analysis


def vhe_sales_merge(*args):
    print('Merging VHE Sales across CA and CRP from DW and Current...')

    dfs = []
    for file in args:
        df = pd.read_csv(file, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
        dfs.append(df)

    df = pd.concat(dfs)
    df.dropna(inplace=True)  # Since i don't know yet which columns are import for future matching, i'll just remove all NaN's. If some of the columns with NaN's are not relevant, then it is better to remove them, in order have more available data
    df.to_csv('sql_db/' + 'BI_VHE_Sales.csv')


def pse_sales_merge(*args):
    print('Merging PSE Sales across CA and CRP from DW and Current...')

    dfs = []
    for file in args:
        df = pd.read_csv(file, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True, dtype={'nlr_code': object, 'Registration_Number': str, 'VAT_Number': object, 'PT_Franchise_Desc': str, 'PT_Model_Desc': str, 'PT_Sales_Type_Service_Level_1_Desc': str, 'PT_Sales_Type_Service_Level_2_Desc': str})
        dfs.append(df)

    df = pd.concat(dfs)
    df.drop(['Anos_Viatura'], axis=1, inplace=True)  # I don't think i'll need this column, as i can just calculate it myself and it has over 27% missing info.
    df.dropna(inplace=True)  # Since i don't know yet which columns are import for future matching, i'll just remove all NaN's. If some of the columns with NaN's are not relevant, then it is better to remove them, in order have more available data
    df.to_csv('sql_db/' + 'BI_PSE_Sales.csv')


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

    vhe_sales = 0
    pse_sales = 1

    if vhe_sales:
        vhe_sales_merge(file1, file2, file5, file6)
    if pse_sales:
        pse_sales_merge(file3, file4, file7, file8)

    print(time.time() - start)


if __name__ == '__main__':
    main()
