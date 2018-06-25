import pandas as pd
import numpy as np
import sys
import csv
import os
import re
import time
import pydotplus
import matplotlib.pyplot as plt
from sklearn import tree
from io import StringIO
from imblearn.over_sampling import RandomOverSampler
from db_tools import value_count_histogram
from test_06 import ClassFit
from test_04 import ohe
from sklearn.tree import export_graphviz
pd.set_option('display.expand_frame_repr', False)
my_dpi = 96


def main():
    start = time.time()

    prov_and_tipo_enc = 0
    group_cols = 1
    target = ['score_class']
    # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']

    oversample = 1
    sales_place_models = 0

    if not sales_place_models:
        stock_optimization(group_cols, oversample, target, prov_and_tipo_enc)
    if sales_place_models:
        stock_optimization_sales_place(group_cols, oversample, target)
    # customer_segmentation()

    print('Running Time: %.2f' % (time.time() - start))


def stock_optimization(group_cols, oversample, target_column, prov_and_tipo_enc):
    targets = ['margem_percentagem', 'Margem', 'stock_days']
    start = time.time()

    if prov_and_tipo_enc:
        dtypes = {'Modelo': str, 'Prov': str, 'Local da Venda': str, 'Tipo Encomenda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}
        cols_to_use = ['Unnamed: 0', 'Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'Tipo Encomenda', 'stock_days', 'Margem', 'margem_percentagem']
        ohe_cols = ['Jantes', 'Cor_Interior', 'Cor_Exterior', 'Local da Venda', 'Modelo', 'Prov', 'Tipo Encomenda']
        if group_cols:
            ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Local da Venda_new', 'Modelo_new', 'Tipo Encomenda_new', 'Prov_new']

    elif not prov_and_tipo_enc:
        dtypes = {'Modelo': str, 'Local da Venda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}
        cols_to_use = ['Unnamed: 0', 'Modelo', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'stock_days', 'Margem', 'margem_percentagem']
        ohe_cols = ['Jantes', 'Cor_Interior', 'Cor_Exterior', 'Local da Venda', 'Modelo']
        if group_cols:
            ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Local da Venda_new', 'Modelo_new']

    if group_cols:
        print('Grouping small count values')
    if prov_and_tipo_enc:
        print('Using Prov and Tipo Encomenda columns')

    df = pd.read_csv('output/' + 'db_baviera_stock_optimization.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    df, targets = db_score_calculation(df, targets)

    if group_cols:
        col_group(df, prov_and_tipo_enc)

    targets += class_creation(df)
    df = ohe(df, ohe_cols)
    df = target_cols_removal(df, target_column, targets)
    df_train_x, df_train_y, df_test_x, df_test_y = dataset_split(df, target_column, oversample)

    tuned_parameters_dt = [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 50], 'max_depth': [3, 5, 7, 9], 'class_weight': ['balanced']}]
    dt = ClassFit(clf=tree.DecisionTreeClassifier)
    dt.grid_search(parameters=tuned_parameters_dt, k=10)
    dt.clf_fit(x=df_train_x, y=df_train_y)

    dt_best = tree.DecisionTreeClassifier(**dt.grid.best_params_)
    dt_best.fit(df_train_x, df_train_y)

    feat_importance = dt_best.feature_importances_

    name = tag(group_cols, prov_and_tipo_enc, target_column)
    feature_importance_graph(list(df_train_x), feat_importance, name)
    feature_importance_csv(list(df_train_x), feat_importance, name)
    performance_evaluation(dt, dt_best, df_train_x, df_train_y, df_test_x, df_test_y, name, start)


def db_score_calculation(df, targets):
    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    df.drop(['stock_days_norm', 'inv_stock_days_norm', 'margem_percentagem_norm'], axis=1, inplace=True)
    targets += ['score']

    return df, targets


def stock_optimization_sales_place(group_cols, oversample, target_column):
    dtypes = {'Modelo': str, 'Local da Venda': str, 'Tipo Encomenda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}

    if group_cols:
        print('Grouping small count values')

    df = pd.read_csv('output/' + 'db_baviera_stock_optimization.csv', encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)

    if group_cols:
        # ohe_cols = ['Prov_new', 'Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Tipo Encomenda_new', 'Local da Venda', 'Modelo']
        ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Modelo_new', 'Local da Venda_new']
        col_group(df)
    elif not group_cols:
        ohe_cols = ['Prov', 'Jantes', 'Cor_Interior', 'Cor_Exterior', 'Tipo Encomenda', 'Local da Venda', 'Modelo']

    df_copy = df
    for value in df['Local da Venda_new'].unique()[:3]:
        print(value)
        df = df_copy.loc[df_copy['Local da Venda_new'] == value]

        class_creation(df)
        df = ohe(df, ohe_cols)
        df = target_cols_removal(df)
        df_train_x, df_train_y, df_test_x, df_test_y = dataset_split(df, target_column, oversample)

        tuned_parameters_dt = [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 50], 'max_depth': [3, 5], 'class_weight': ['balanced']}]
        dt = ClassFit(clf=tree.DecisionTreeClassifier)
        dt.grid_search(parameters=tuned_parameters_dt, k=10)
        dt.clf_fit(x=df_train_x, y=df_train_y)

        dt_best = tree.DecisionTreeClassifier(**dt.grid.best_params_)
        dt_best.fit(df_train_x, df_train_y)
        decision_tree_plot(dt_best, 'output/', value, df_train_x, oversample)

        feat_importance = dt_best.feature_importances_
        performance_evaluation(dt, dt_best, df_train_x, df_train_y, df_test_x, df_test_y)
        feature_importance_graph(list(df_train_x), feat_importance)


def customer_segmentation():
    df = pd.read_csv('sql_db/' + 'DataSet_Customer_Segmentation.csv', delimiter=';', index_col=0, header=None, dtype={3: str, 4: str})
    # renames = {1: 'cliente_fatura', 2: 'empresa', 3: 'data_fatura', 4: 'marca', 5: 'modelo', 6: 'km', 7: 'dt_entrada', 8:'dt_matricula', 9:'anos_viatura', 10: 'soldbygsc', 11:'tipo_venda_nivel_1', 12:'tipo_venda_nivel_1', 13:'departamento', 14:'valor_faturado', 15:'visita_em garantia', 16: 'cm_check', 21: 'visita_em_cm'}
    # cols_to_drop = [17, 18, 19, 20]
    # df.rename(renames, axis=1, inplace=True)
    # df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.head())


def decision_tree_plot(clf, output_dir, tag, df_train_x, oversample):
    print('Plotting Decision Tree...')
    file_name = str(tag) + '_decision_tree'
    if oversample:
        file_name += '_oversampled'

    dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, feature_names=list(df_train_x), class_names=['0', '1', '2', '3'], special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(output_dir + file_name + '.pdf')


def target_cols_removal(df, target, targets_to_remove):

    targets_to_remove.remove(target[0])
    for value in targets_to_remove:
        df.drop(value, axis=1, inplace=True)

    # df.drop('margem_percentagem', axis=1, inplace=True)
    # df.drop('stock_days', axis=1, inplace=True)
    # df.drop('Margem', axis=1, inplace=True)
    # df.drop('margem_class1', axis=1, inplace=True)

    return df


def oversample_data(train_x, train_y):

    ros = RandomOverSampler(random_state=42)
    train_x_resampled, train_y_resampled = ros.fit_sample(train_x, train_y)

    return pd.DataFrame(np.atleast_2d(train_x_resampled), columns=list(train_x)), pd.Series(train_y_resampled.tolist())


def col_group(df, prov_and_tipo_enc):
    # Cor_Exterior
    column_grouping(df, column='Cor_Exterior', values_to_keep=['preto', 'cinzento', 'branco', 'azul'])
    # value_count_histogram(df, 'Cor_Exterior', 'cor_exterior_before')
    # value_count_histogram(df, 'Cor_Exterior_new', 'cor_exterior_after')

    # Cor_Interior
    column_grouping(df, column='Cor_Interior', values_to_keep=['preto', 'antracite', 'dakota', 'antracite/cinza/preto'])
    # value_count_histogram(df, 'Cor_Interior', 'cor_interior_before')
    # value_count_histogram(df, 'Cor_Interior_new', 'cor_interior_after')

    # Jantes
    column_grouping(df, column='Jantes', values_to_keep=['standard', '17', '18', '19'])
    # value_count_histogram(df, 'Jantes', 'jantes_before')
    # value_count_histogram(df, 'Jantes_new', 'jantes_after')

    # Modelo
    model_grouping(df)
    # value_count_histogram(df, 'Modelo', 'modelo_before')
    # value_count_histogram(df, 'Modelo_new', 'modelo_after')

    # Local da Venda
    sales_place_grouping(df)
    # value_count_histogram(df, 'Local da Venda', 'local_da_venda_before')
    # value_count_histogram(df, 'Local da Venda_new', 'local_da_venda_after')

    if prov_and_tipo_enc:
        # Prov
        column_grouping(df, column='Prov', values_to_keep=['Novos', 'Demonstração'])
        # value_count_histogram(df, 'Prov', 'prov_before')
        # value_count_histogram(df, 'Prov_new', 'prov_after')

        # Tipo Encomenda
        column_grouping(df, column='Tipo Encomenda', values_to_keep=['Enc Client Final', 'Encomenda Stock'])
        # value_count_histogram(df, 'Tipo Encomenda', 'tipo_encomenda_before')
        # value_count_histogram(df, 'Tipo Encomenda_new', 'tipo_encomenda_after')

        df.drop('Prov', axis=1, inplace=True)
        df.drop('Tipo Encomenda', axis=1, inplace=True)

    df.drop('Cor_Exterior', axis=1, inplace=True)
    df.drop('Cor_Interior', axis=1, inplace=True)
    df.drop('Jantes', axis=1, inplace=True)
    df.drop('Modelo', axis=1, inplace=True)
    df.drop('Local da Venda', axis=1, inplace=True)


def sales_place_grouping(df):
    # print(df['Local da Venda'].unique())

    # 1st grouping:
    # norte_group = ['DCC - Feira', 'DCG - Gaia', 'DCV - Coimbrões', 'DCN-Porto', 'DCN-Porto Mini', 'DCC - Aveiro', 'DCG - Gaia Mini']
    # norte_group_used = ['DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
    # center_group = ['DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi']
    # center_group_used = ['DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad']
    # south_group = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro']
    # south_group_used = ['DCA -Portimão Usados']

    # 2nd grouping:
    norte = ['DCC - Feira', 'DCG - Gaia', 'DCV - Coimbrões', 'DCN-Porto', 'DCN-Porto Mini', 'DCC - Aveiro', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
    centro = ['DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad']
    sul = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados']

    motorcycles = ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']
    unknown_group = ['DCS-Expo Frotas Flee', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Busi', 'DCS-V Especiais BMW']

    # groups = [norte_group, norte_group_used, center_group, center_group_used, south_group, south_group_used, motorcycles, unknown_group]
    # groups_name = ['norte', 'norte_usados', 'centro', 'centro_usados', 'sul', 'sul_usados', 'motos', 'unknown']
    groups = [norte, centro, sul, motorcycles, unknown_group]
    groups_name = ['norte', 'centro', 'sul', 'motos', 'unknown']
    for group in groups:
        for dc in group:
            df.loc[df['Local da Venda'] == dc, 'Local da Venda_new'] = groups_name[groups.index(group)]

    # print(df[['Local da Venda', 'Local da Venda New']])
    return df


def model_grouping(df):

    s1 = ['S1 3p', 'S1 5p']
    s2 = ['S2 Active Tourer', 'S2 Cabrio', 'S2 Gran Tourer', 'S2 Coupé']
    s3 = ['S3 Touring', 'S3 Gran Turismo', 'S3 Berlina']
    s4 = ['S4 Gran Coupé', 'S4 Coupé']
    s5 = ['S5 Touring', 'S5 Limousine', 'S5 Gran Turismo', 'S5 Berlina']
    s6 = ['S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé']
    s7 = ['S7 Berlina', 'S7 L Berlina']
    x1 = ['X1']
    x2 = ['X2 SAC']
    x3 = ['X3 SUV']
    x4 = ['X4 SUV']
    x5 = ['X5 SUV', 'X5 M']
    x6 = ['X6' 'X6 M']
    z4 = ['Z4 Roadster']
    motos = ['Série C', 'Série F', 'Série K', 'Série R']

    groups = [s1, s2, s3, s4, s5, s6, s7, x1, x2, x3, x4, x5, x6, z4, motos]
    groups_name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Z4', 'Motociclos']
    for group in groups:
        for dc in group:
            df.loc[df['Modelo'] == dc, 'Modelo_new'] = groups_name[groups.index(group)]

    return df


def column_grouping(df, column, values_to_keep):

    all_values = df[column].value_counts().index
    values_to_change = [x for x in all_values if x not in values_to_keep]
    df[column + '_new'] = df[column]

    for value in values_to_change:
        df.loc[df[column] == value, column + '_new'] = 'outros'


def class_creation(df):
    # Business Values
    df.loc[df['stock_days'] <= 16, 'stock_class1'] = 0
    df.loc[(df['stock_days'] > 16) & (df['stock_days'] <= 60), 'stock_class1'] = 1
    df.loc[(df['stock_days'] > 60) & (df['stock_days'] <= 100), 'stock_class1'] = 2
    df.loc[(df['stock_days'] > 100) & (df['stock_days'] <= 200), 'stock_class1'] = 3
    df.loc[df['stock_days'] > 200, 'stock_class1'] = 4

    # Quartile Limits
    df.loc[df['stock_days'] <= 16, 'stock_class2'] = 0
    df.loc[(df['stock_days'] > 16) & (df['stock_days'] <= 29), 'stock_class2'] = 1
    df.loc[(df['stock_days'] > 29) & (df['stock_days'] <= 117), 'stock_class2'] = 2
    df.loc[df['stock_days'] > 117, 'stock_class2'] = 3

    # Quartile Limits
    df.loc[df['margem_percentagem'] <= 0.1, 'margem_class1'] = 3
    df.loc[(df['margem_percentagem'] > 0.1) & (df['margem_percentagem'] <= 0.206), 'margem_class1'] = 2
    df.loc[(df['margem_percentagem'] > 0.206) & (df['margem_percentagem'] <= 0.349), 'margem_class1'] = 1
    df.loc[df['margem_percentagem'] > 0.349, 'margem_class1'] = 0

    # 4 Classes:
    df.loc[(df['score'] >= 0) & (df['score'] < 0.25), 'score_class'] = 0
    df.loc[(df['score'] >= 0.25) & (df['score'] < 0.50), 'score_class'] = 1
    df.loc[(df['score'] >= 0.50) & (df['score'] <= 0.75), 'score_class'] = 2
    df.loc[(df['score'] >= 0.75), 'score_class'] = 3

    new_targets_created = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']
    return new_targets_created


def dataset_split(df, target, oversample):
    print('Splitting dataset...')

    # df_train, df_test = train_test_split(df, test_size=0.2)
    df_train = df.head(4317)  # 80%
    df_test = df.tail(1080)  # 20%

    df_train_y = df_train[target]
    df_train_x = df_train.drop(target, axis=1)

    df_test_y = df_test[target]
    df_test_x = df_test.drop(target, axis=1)

    if oversample:
        print('Oversampling small classes...')
        df_train_x, df_train_y = oversample_data(df_train_x, df_train_y)

    return df_train_x, df_train_y, df_test_x, df_test_y


def tag(group_cols, prov_and_tipo_enc, target):

    file_name = '_target' + str(target[0])
    if group_cols:
        file_name += '_group_cols'
    if prov_and_tipo_enc:
        file_name += '_prov_and_type'

    return file_name


def feature_importance_csv(features, feature_importance, name):
    indices = np.argsort(feature_importance)[::-1]

    file_name = 'feature_importance_'
    file_name += name

    if os.path.isfile('output/' + file_name + '.csv'):
        os.remove('output/' + file_name + '.csv')

    with open('output/' + file_name + '.csv', 'a', newline='') as csvfile:
        fieldnames = ['Rank', 'Feature', 'Importance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for f in range(len(features)):
            writer.writerow({'Rank': f + 1, 'Feature': features[indices[f]], 'Importance': feature_importance[indices[f]]})


def feature_importance_graph(features, feature_importance, name):
    print('Plotting Feature Importance...')

    file_name = 'feature_importance_'
    file_name += name

    if os.path.isfile('output/' + file_name + 'png'):
        os.remove('output/' + file_name + '.png')

    d = {'feature': features, 'importance': feature_importance}
    feat_importance_df = pd.DataFrame(data=d)
    feat_importance_df.sort_values(ascending=False, by='importance', inplace=True)

    # top_features = feat_importance_df[feat_importance_df['importance'] > 0.01]
    top_features = feat_importance_df.head(10)
    print(top_features)

    # plt.figure()
    plt.subplots(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    plt.title('Top 10 - Feature Importance')
    plt.bar(range(top_features.shape[0]), top_features['importance'], color='r', align='center', zorder=3)
    plt.xticks(range(top_features.shape[0]), top_features['feature'], rotation=20)
    plt.xlim([-1, top_features.shape[0]])
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.grid()

    plt.savefig('output/' + file_name + '.png')
    # plt.show()
    plt.clf()
    plt.close()


def performance_evaluation(model, best_model, train_x, train_y, test_x, test_y, name, start):

    prediction_trainer = best_model.predict(train_x)
    model.grid_performance(prediction=prediction_trainer, y=train_y)
    print('Train:')
    print('Micro:', model.micro, '\n', 'Macro:', model.macro, '\n', 'Accuracy:', model.accuracy, '\n', 'Class Report', '\n', model.class_report)
    prediction_test = best_model.predict(test_x)
    model.grid_performance(prediction=prediction_test, y=test_y)
    print('Test:')
    print('Micro:', model.micro, '\n', 'Macro:', model.macro, '\n', 'Accuracy:', model.accuracy, '\n', 'Class Report', '\n', model.class_report)

    class_report_csv(model.class_report, name)
    performance_metrics_csv(model.micro, model.macro, model.accuracy, name, start)


def performance_metrics_csv(micro, macro, accuracy, name, start):

    file_name = 'performance_evaluation_'
    file_name += name

    if os.path.isfile('output/' + file_name + '.csv'):
        os.remove('output/' + file_name + '.csv')

    # header = 0
    with open('output/' + file_name + '.csv', 'a', newline='') as csvfile:
        fieldnames = ['micro_f1', 'macro_f1', 'accuracy', 'running_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'micro_f1': micro, 'macro_f1': macro, 'accuracy': accuracy, 'running_time': (time.time() - start)})


def class_report_csv(report, name):

    file_name = 'class_report_'
    file_name += name

    if os.path.isfile('output/' + file_name + '.csv'):
        os.remove('output/' + file_name + '.csv')

    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)
    report_df.to_csv('output/' + file_name + '.csv')


if __name__ == '__main__':
    main()
