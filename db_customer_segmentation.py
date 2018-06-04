import pandas as pd
import numpy as np
# -*- coding: 'latin-1' -*-
pd.set_option('display.expand_frame_repr', False)
from db_analysis import null_analysis

file1 = 'sql_db/' + 'BI_CA_VHE_Sales.csv'
file2 = 'sql_db/' + 'BI_CRP_VHE_Sales.csv'
file3 = 'sql_db/' + 'BI_CA_PSE_Sales.csv'
file4 = 'sql_db/' + 'BI_CRP_PSE_Sales.csv'
file5 = 'sql_db/' + 'BI_DW_CA_VHE_Sales.csv'
file6 = 'sql_db/' + 'BI_DW_CRP_VHE_Sales.csv'
file7 = 'sql_db/' + 'BI_DW_CA_PSE_Sales.csv'
file8 = 'sql_db/' + 'BI_DW_CRP_PSE_Sales.csv'
file9 = 'sql_db/' + 'BSI_2018053000.txt'  # Contratos de Manutenção BMW/Mini

df1 = pd.read_csv(file1, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
df2 = pd.read_csv(file2, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
df3 = pd.read_csv(file3, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
df4 = pd.read_csv(file4, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True, dtype={'nlr_code': object, 'Registration_Number': str, 'VAT_Number': object, 'PT_Franchise_Desc': str, 'PT_Model_Desc': str, 'PT_Sales_Type_Service_Level_1_Desc': str, 'PT_Sales_Type_Service_Level_2_Desc': str})
df5 = pd.read_csv(file5, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
df6 = pd.read_csv(file6, delimiter=';', encoding='latin-1', parse_dates=['slr_document_date'], infer_datetime_format=True)
df7 = pd.read_csv(file7, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
df8 = pd.read_csv(file8, delimiter=';', encoding='latin-1', parse_dates=['SLR_Document_Date', 'Vehicle_In_Date', 'Registration_Date'], infer_datetime_format=True)
# df9 = pd.read_csv(file9, delimiter=',', encoding='latin-1', header=None, parse_dates=[4, 5], infer_datetime_format=True)



