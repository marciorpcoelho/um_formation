#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
import sys
from db_tools import null_analysis
import time
import os
from db_tools import save_csv
pd.set_option('display.expand_frame_repr', False)

'''
    File name: baviera_option_extraction.py
    Author: Márcio Coelho
    Date created: 30/05/2018
    Date last modified: 01/06/2018
    Python Version: 3.6
'''


def db_creation(df):
    df['Opcional'] = df['Opcional'].str.lower()
    df['Cor'] = df['Cor'].str.lower()
    df['Interior'] = df['Interior'].str.lower()
    # df = df[df['Opcional'] != 'preço base']
    df = df[df['Opcional'] != 'preço de venda']
    df['Modelo'] = df['Modelo'].str.replace(' - não utilizar', '')

    df = symbol_replacer(df)

    df['Navegação'], df['Sensores'], df['Cor_Interior'], df['Caixa Auto'], df['Cor_Exterior'], df['Jantes'] = 0, 0, 0, 0, 0, 0  # New Columns
    colors_pt = ['preto', 'branco', 'azul', 'vermelho', 'cinza', 'cinzento', 'prateado', 'prata', 'amarelo', 'laranja', 'castanho', 'dourado', 'antracit', 'antracite/preto', 'antracite/cinza/preto', 'antracito', 'dakota', 'antracite', 'antracite/vermelho/preto', 'oyster/preto', 'prata/preto/preto', 'âmbar/preto/pr', 'terra', 'preto/laranja', 'cognac/preto', 'bronze', 'beige', 'veneto/preto', 'zagora/preto', 'mokka/preto', 'taupe/preto', 'sonoma/preto', 'preto/preto']
    colors_en = ['black', 'white', 'blue', 'red', 'grey', 'silver', 'orange', 'green', 'bluestone', 'aqua', 'burgundy', 'anthrazit', 'truffle', 'brown', 'oyster', 'tobacco', 'jatoba', 'storm', 'champagne', 'cedar', 'silverstone', 'chestnut', 'kaschmirsilber', 'oak', 'mokka']
    # Que cor é esta? ['sparkling', 'storm', 'metalizada', 'brilhante']

    df_grouped = df.groupby('Nº Stock')
    for key, group in df_grouped:
        ### Navegação/Sensor/Transmissão
        for line_options in group['Opcional']:
            tokenized_options = nltk.word_tokenize(line_options)
            if 'navegação' in tokenized_options:
                df.loc[df['Nº Stock'] == key, 'Navegação'] = 1
            if 'sensor' and 'dianteiros' in tokenized_options:
                df.loc[df['Nº Stock'] == key, 'Sensores'] = 1
            if 'transmissão' and 'automática' in tokenized_options:
                df.loc[df['Nº Stock'] == key, 'Caixa Auto'] = 1

        ### Cor Exterior
        line_color = group['Cor'].head(1).values[0]
        tokenized_color = nltk.word_tokenize(line_color)
        color = [x for x in colors_pt if x in tokenized_color]
        if not color:
            color = [x for x in colors_en if x in tokenized_color]
        if not color:
            if tokenized_color == ['pintura', 'bmw', 'individual'] or ['hp', 'motorsport', ':', 'branco/azul/vermelho', '``', 'racing', "''"] or ['p0b58']:
                continue
            else:
                print(tokenized_color)
                sys.exit('Error: Color Not Found')
        if len(color) > 1:  # Fixes cases such as 'white silver'
            color = [color[0]]
        color = color * group.shape[0]
        df.loc[df['Nº Stock'] == key, 'Cor_Exterior'] = color

        ### Cor Interior
        line_interior = group['Interior'].head(1).values[0]
        tokenized_interior = nltk.word_tokenize(line_interior)

        color_interior = [x for x in colors_pt if x in tokenized_interior]

        if not color_interior:
            color_interior = [x for x in colors_en if x in tokenized_interior]

        if 'truffle' in tokenized_interior:
            # print('truffle found!', tokenized_interior)
            color_interior = ['truffle']

        if 'banco' and 'standard' in tokenized_interior:
            color_interior = ['preto']

        if not color_interior:
            continue

        if len(color_interior) > 1:
            # print('Too Many Colors:', tokenized_interior, color_interior)
            if 'dakota' in tokenized_interior:
                color_interior = ['dakota']
            if 'nevada' in tokenized_interior:
                color_interior = ['nevada']
            # print('Changed to:', color_interior)

        color_interior = color_interior * group.shape[0]
        df.loc[df['Nº Stock'] == key, 'Cor_Interior'] = color_interior

        ### Jantes
        for line_options in group['Opcional']:
            tokenized_jantes = nltk.word_tokenize(line_options)
            for value in range(15, 21):
                if str(value) in tokenized_jantes:
                    jantes_size = [str(value)] * group.shape[0]
                    df.loc[df['Nº Stock'] == key, 'Jantes'] = jantes_size

        ## Modelo
        line_modelo = group['Modelo'].head(1).values[0]
        tokenized_modelo = nltk.word_tokenize(line_modelo)
        if tokenized_modelo[0] == 'Série':
            df.loc[df['Modelo'] == line_modelo, 'Modelo'] = ' '.join(tokenized_modelo[:2])
        else:
            df.loc[df['Modelo'] == line_modelo, 'Modelo'] = ' '.join(tokenized_modelo[:-3])

    df.loc[df['Jantes'] == 0, 'Jantes'] = 'standard'
    return df


def symbol_replacer(df):

    df['Interior'] = df['Interior'].str.replace('|', '/')
    df['Cor'] = df['Cor'].str.replace('|', '')
    df['Interior'] = df['Interior'].str.replace('ind.', '')

    return df


def db_color_replacement(df):
    color_types = ['Cor_Interior', 'Cor_Exterior']
    colors_to_replace = {'black': 'preto', 'white': 'branco', 'blue': 'azul', 'red': 'vermelho', 'grey': 'cinzento', 'silver': 'prateado', 'orange': 'laranja', 'green': 'verde', 'anthrazit': 'antracite', 'antracit': 'antracite', 'brown': 'castanho', 'antracito': 'antracite', 'âmbar/preto/pr': 'ambar/preto/preto', 'beige': 'bege', 'kaschmirsilber': 'cashmere'}

    unknown_ext_colors = df[df['Cor_Exterior'] == 0]['Cor'].unique()
    unknown_int_colors = df[df['Cor_Interior'] == 0]['Interior'].unique()
    print('Unknown Exterior Colors:', unknown_ext_colors, ', Removed', df[df['Cor_Exterior'] == 0].shape[0], 'lines in total, corresponding to ', df[df['Cor_Exterior'] == 0]['Nº Stock'].nunique(), 'vehicles')  # 49 lines removed, 3 vehicles
    print('Unknown Interior Colors:', unknown_int_colors, ', Removed', df[df['Cor_Interior'] == 0].shape[0], 'lines in total, corresponding to ', df[df['Cor_Interior'] == 0]['Nº Stock'].nunique(), 'vehicles')  # 2120 lines removed, 464 vehicles

    for color_type in color_types:
        df[color_type] = df[color_type].replace(colors_to_replace)
        df.drop(df[df[color_type] == 0].index, axis=0, inplace=True)

    return df


def db_score_calculation(df):
    df['stock_days'] = (df['Data Venda'] - df['Data Compra']).dt.days
    # df['margem_norm'] = (df['Margem'] - df['Margem'].min()) / (df['Margem'].max() - df['Margem'].min())
    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df_grouped = df.groupby('Nº Stock')
    for key, group in df_grouped:
        # print(key, '\n', group)
        total = group['Custo'].sum()
        # print('\n', total)
        df.loc[df['Nº Stock'] == key, 'margem_percentagem'] = group['Margem'] / total

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())

    # df['score_old'] = df['inv_stock_days_norm'] * df['margem_norm']
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    return df


def db_duplicate_removal(df):
    cols_to_drop = ['CdCor', 'Cor', 'CdInt', 'Interior', 'Versão', 'Opcional', 'A', 'S', 'Custo', 'Data Compra', 'Data Venda', 'Vendedor', 'Canal de Venda']
    # Will probably need to also remove: stock_days, stock_days_norm, and one of the scores
    df = df.drop_duplicates(subset='Nº Stock')
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.index = range(df.shape[0])

    return df


def main():
    start = time.time()
    print('Creating DB...')

    # full_db = 'output/' + 'db_full_baviera.csv'
    full_db = 'output/' + 'full_testing.csv'
    # stock_opt_db = 'output/' + 'db_baviera_stock_optimization.csv'
    stock_opt_db = 'output/' + 'testing.csv'
    input_file = 'sql_db/' + 'Opcionais Baviera.csv'
    # input_file = 'output/' + 'small_test.csv'

    if os.path.isfile(full_db):
        os.remove(full_db)

    df = pd.read_csv(input_file, delimiter=';', parse_dates=['Data Compra', 'Data Venda'], infer_datetime_format=True, decimal=',').dropna()
    df_initial = db_creation(df)
    df_second_step = db_color_replacement(df_initial)
    df_third_step = db_score_calculation(df_second_step)
    df_final = db_duplicate_removal(df_third_step)

    sel_cols = ['Modelo', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'Tipo Encomenda', 'stock_days', 'Margem', 'margem_percentagem']
    # df_final[['Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'Tipo Encomenda', 'stock_days', 'Margem', 'margem_percentagem']].to_csv(stock_opt_db)
    save_csv(df_final[sel_cols], stock_opt_db)

    # if os.path.isfile(full_db):
    #     os.remove(full_db)
    # df_final.to_csv(full_db)
    save_csv(df_final, full_db)

    print('Runnning time: %.2f' % (time.time() - start))


if __name__ == '__main__':
    main()
