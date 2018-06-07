import pandas as pd
import time
import numpy as np
import re
import sys
from db_analysis import null_analysis
# -*- coding: 'latin-1' -*-
pd.set_option('display.expand_frame_repr', False)


def main():
    warranty_years_ca = 5
    warranty_years_crp = 2

    dtypes = {'nlr_code': int, 'slr_account': str, 'kms': int, 'customer': str, 'slr_document_account': str}
    parse_dates = ['vehicle_in_date', 'registration_date', 'cm_date_start', 'cm_date_end']

    df = pd.read_csv('sql_db/' + 'db.csv', index_col=0, parse_dates=parse_dates, dtype=dtypes)
    print(df.head(50))
    print(df[df['customer'] == '2.0'].shape)


def warranty_visits(db, pse_sales, vhe_sales, cm_toyota_lexus, cm_bmw_mini, warranty_years_ca, warranty_years_crp):
    start = time.time()
    print('Number of warranty visits...')

    vhe_sold_by_gsc = pse_sales[pse_sales['soldbygsc'] == 1]['registration_number'].unique()
    warranties_in, warranties_out = [], []
    # for registration_number in db.index.get_level_values(1):
    for registration_number in vhe_sold_by_gsc:
        start = time.time()
        vhe = vhe_sales[vhe_sales['registration_number'] == registration_number]
        if vhe['nlr_code'].head(1).values == 101:
            warranty_years = warranty_years_ca
        else:
            warranty_years = warranty_years_crp
        pse = pse_sales[pse_sales['registration_number'] == registration_number]

        # print('\n', vhe, '\n', vhe.shape, '\n', pse, '\n', pse.shape)

        try:
            date_sold = list(vhe['slr_document_date'].values)[0]
            date_warranty_limit = date_sold + np.timedelta64(365 * warranty_years, 'D')

            pse = pse[pse['slr_document_date'] >= date_sold]
            warranty_count_in = pse[pse['slr_document_date'] <= date_warranty_limit].shape[0]
            warranty_count_out = pse[pse['slr_document_date'] > date_warranty_limit].shape[0]

        except IndexError:
            # print('empty date')
            warranty_count_in, warranty_count_out = 'NaN', 'NaN'

        # print(warranty_count_in, warranty_count_out)
        print('Running time: %.2f' % (time.time() - start), 'seconds')
        warranties_in.append(warranty_count_in)
        warranties_out.append(warranty_count_out)


    print(len(warranties_in), len(warranties_out))

    db['visits_in_warranty'] = warranties_in
    db['visits_out_warranty'] = warranties_out

    print('Running time: %.2f' % (time.time() - start), 'seconds')


if __name__ == '__main__':
    main()

# print(time.time() - start)

