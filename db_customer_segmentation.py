import pandas as pd
import time
import numpy as np
import re
import sys
from db_analysis import null_analysis, save_csv
from datetime import datetime, timedelta
# -*- coding: 'latin-1' -*-
pd.set_option('display.expand_frame_repr', False)


def main():
    start = time.time()
    warranty_years_ca = 5
    warranty_years_crp = 2

    dtypes = {'nlr_code': int, 'slr_account': str, 'kms': int, 'customer': str, 'slr_document_account': str}
    parse_dates = ['vehicle_in_date', 'registration_date_pse', 'registration_date_vhe', 'cm_date_start', 'cm_date_end']

    df = pd.read_csv('sql_db/' + 'db.csv', index_col=0, parse_dates=parse_dates, dtype=dtypes)

    db_creation(df, warranty_years_ca, warranty_years_crp)

    print('Runnning time: %.2f' % (time.time() - start), 'seconds')


def time_last_visit(x, current_datetime):
    time_since_last_visit = current_datetime - x['vehicle_in_date'].tail(1)
    time_since_last_visit_years = time_since_last_visit.dt.days / 365.25

    if list(time_since_last_visit_years.values)[0] <= 2:
        return 0
    elif list(time_since_last_visit_years.values)[0] > 2:
        return 1
    else:
        return 'NULL'

    # return pd.Series(all)


def regular_percent_pse(x, current_datetime):

    time_since_bought = current_datetime - x['registration_date_pse'].head(1)
    time_since_bought_years = time_since_bought.dt.days / 365.25
    expected_visits = list(time_since_bought_years)[0] - 1
    if str(expected_visits) == 'nan':
        return 'NULL'
    if expected_visits < 1:
        return '<2years'
    if expected_visits >= 1:
        regular_percentage = (x.shape[0] * 1. / expected_visits * 100)
        if regular_percentage > 100:  # Case when a client comes more than the expected number of times
            regular_percentage = 100
            # print(expected_visits, x.shape[0], regular_percentage)
        return regular_percentage

    # return regular_percentage


def regular_percent_vhe(x, current_datetime):

    time_since_bought = current_datetime - x['registration_date_vhe'].head(1)
    time_since_bought_years = time_since_bought.dt.days / 365.25
    expected_visits = list(time_since_bought_years)[0] - 1
    if str(expected_visits) == 'nan':
        return 'NULL'
    if expected_visits < 1:
        return '<2years'
    if expected_visits >= 1:
        regular_percentage = (x.shape[0] * 1. / expected_visits * 100)
        if regular_percentage > 100:  # Case when a client comes more than the expected number of times
            regular_percentage = 100
            # print(expected_visits, x.shape[0], regular_percentage)
        return regular_percentage


def db_creation(df, warranty_years_ca, warranty_years_crp):
    print('Creating DB fields...')
    start = time.time()
    current_datetime = datetime.now()

    df['warranty_visit_pse'], df['warranty_visit_vhe'], df['contract_visit'], df['regular_percentage_pse'], df['regular_percentage_vhe'], df['abandoned'] = 0, 0, 0, 0, 0, 0
    df.sort_values(by=['vehicle_in_date'], inplace=True)
    df_grouped = df.groupby(['customer', 'registration_number'], as_index=False)

    ### Predefined values:
    df.loc[df['registration_date_pse'].isnull(), 'warranty_visit_pse'], df.loc[df['registration_date_vhe'].isnull(), 'warranty_visit_vhe'] = 'NULL', 'NULL'
    df.loc[df['vehicle_in_date'].isnull(), ['warranty_visit_pse', 'warranty_visit_vhe', 'contract_visit', 'abandoned']] = 'NULL'
    df.loc[df['cm_date_end'].isnull(), 'contract_visit'] = 'NULL'
    print('1st step done at %.2f' % (time.time() - start))

    ### Warranty Visit?
    df_ca = df[df['nlr_code'] == 101]
    df_ca.loc[df_ca['vehicle_in_date'] < (df_ca['registration_date_pse'] + np.timedelta64(365 * warranty_years_ca, 'D')), 'warranty_visit_pse'] = 1
    df_ca.loc[df_ca['vehicle_in_date'] < (df_ca['registration_date_vhe'] + np.timedelta64(365 * warranty_years_ca, 'D')), 'warranty_visit_vhe'] = 1

    df_crp = df[df['nlr_code'] == 701]
    df_crp.loc[df_crp['vehicle_in_date'] < (df_crp['registration_date_pse'] + np.timedelta64(365 * warranty_years_crp, 'D')), 'warranty_visit_pse'] = 1
    df_crp.loc[df_crp['vehicle_in_date'] < (df_crp['registration_date_vhe'] + np.timedelta64(365 * warranty_years_crp, 'D')), 'warranty_visit_vhe'] = 1

    df_all = pd.concat([df_ca, df_crp])
    print('2nd step done at %.2f' % (time.time() - start))

    ### Contract Visit?
    # df_all.loc[df_all['vehicle_in_date'] < df_all['cm_date_end'], 'contract_visit'] = 1
    df_all.loc[(df_all['vehicle_in_date'] <= df_all['cm_date_end']) & (df_all['anos_viatura'] <= df_all['cm_years']) & (df_all['kms'] <= df_all['cm_km']), 'contract_visit'] = 1
    print('3rd step done at %.2f' % (time.time() - start))

    ### Abandonded?
    something = df_grouped.apply(time_last_visit, current_datetime=current_datetime)
    df_all = df_all.merge(something.to_frame(), on=['customer', 'registration_number'])
    df_all.drop(['abandoned'], axis=1, inplace=True)
    df_all = df_all.rename(columns={0: 'abandoned'})
    print('4th step done at %.2f' % (time.time() - start))

    # ### Regular Percentage? - PSE
    # something_2 = df_grouped.apply(regular_percent_pse, current_datetime=current_datetime)
    # df_all = df_all.merge(something_2.to_frame(), on=['customer', 'registration_number'])
    # df_all.drop(['regular_percentage_pse'], axis=1, inplace=True)
    # df_all = df_all.rename(columns={0: 'regular_percentage_pse'})
    # print('5th step done at %.2f' % (time.time() - start))
    #
    # ### Regular Percentage? - VHE
    # something_3 = df_grouped.apply(regular_percent_vhe, current_datetime=current_datetime)
    # df_all = df_all.merge(something_3.to_frame(), on=['customer', 'registration_number'])
    # df_all.drop(['regular_percentage_vhe'], axis=1, inplace=True)
    # df_all = df_all.rename(columns={0: 'regular_percentage_vhe'})
    # print('6th step done at %.2f' % (time.time() - start))

    save_csv(df_all, 'output/' + 'db_customer_segmentation_short.csv')

if __name__ == '__main__':
    main()

