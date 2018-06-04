import pandas as pd
# import numpy as np
import time
from db_analysis import null_analysis
# -*- coding: 'latin-1' -*-
pd.set_option('display.expand_frame_repr', False)

start = time.time()


# df9 = pd.read_csv(file9, delimiter=',', encoding='latin-1', header=None, parse_dates=[4, 5], infer_datetime_format=True)

# vhe_sales = pd.read_csv('sql_db/' + 'BI_VHE_Sales.csv', index_col=0, delimiter=',', encoding='utf-8', parse_dates=['slr_document_date'], infer_datetime_format=True)
# pse_sales = pd.read_csv('sql_db/' + 'BI_PSE_Sales.csv', index_col=0, delimiter=',', encoding='utf-8', parse_dates=['slr_document_date', 'vehicle_in_date', 'registration_date'], infer_datetime_format=True, dtype={'nlr_code': object, 'Registration_Number': str, 'VAT_Number': object, 'PT_Franchise_Desc': str, 'PT_Model_Desc': str, 'PT_Sales_Type_Service_Level_1_Desc': str, 'PT_Sales_Type_Service_Level_2_Desc': str, 'slr_document_account': str})
cm_bmw_mini = pd.read_csv('sql_db/' + 'BSI_2018053000.txt', delimiter=',', encoding='latin-1', header=None, parse_dates=[4, 5], infer_datetime_format=True)
# cm_toyota_lexus = pd.read_csv('sql_db/' + 'cm_toyota_lexus.csv', delimiter=',', index_col=0, parse_dates=['cm_date_start', 'cm_date_end', 'cm_date_removal'])

# print(vhe_sales[vhe_sales['nlr_code'] == 701].head())
# print(cm_bmw_mini.head())

df = pd.read_csv('sql_db/' + 'BI_CRP_VHE_Sales.csv', delimiter=';', encoding='latin-1')
print(df.head())
print(cm_bmw_mini.head())


# pse_sales_grouped = pse_sales.groupby(['slr_document_account', 'registration_number'])
# for key, group in pse_sales_grouped:
#     print(key, '\n', group)



print(time.time() - start)

