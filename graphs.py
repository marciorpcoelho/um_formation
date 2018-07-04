import pandas as pd
import numpy as np
import itertools
from db_tools import null_analysis
import matplotlib.pyplot as plt
from db_tools import null_analysis, save_fig
from scipy.stats import pearsonr
from big_data_formation_module1 import db_score_calculation, column_grouping
from matplotlib.ticker import FormatStrFormatter
pd.set_option('display.expand_frame_repr', False)
my_dpi = 96


def main():
    input_dir = 'output/'
    target = 'score_class'
    # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']
    targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']
    oversample = 0

    # performance_evaluation_plot(input_dir)
    # performance_evaluation_plot_all(input_dir, targets)
    # average_score(input_dir)
    # score_distribution()
    classification_report_plot(input_dir, oversample)


def classification_report_plot(input_dir, oversample):

    f, ax = plt.subplots(3, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    models = ['dt', 'rf', 'lr', 'knn', 'svm', 'ab', 'gc']
    models_name = ['Dec. Tree', 'Rand. Forest', 'Log Reg', 'KNN', 'SVM', 'Adaboost', 'Gradient']
    for model in models:
        model_label = models_name[models.index(model)]
        if oversample:
            cr = pd.read_csv(input_dir + 'class_report_classification_target_score_class_' + str(model) + '_oversample.csv', index_col=0)
            pe = pd.read_csv(input_dir + 'performance_evaluation_classification_target_score_class_' + str(model) + '_oversample.csv')
        if not oversample:
            cr = pd.read_csv(input_dir + 'class_report_classification_target_score_class_' + str(model) + '.csv', index_col=0)
            pe = pd.read_csv(input_dir + 'performance_evaluation_classification_target_score_class_' + str(model) + '.csv')

        ax[0].plot(pe.loc[0, :][:-1].values, label=model_label)
        ax[1].plot(cr.loc['avg/total', :][:-1].values)
        ax[2].plot(cr.loc[:, 'recall'][:-1].values)

    ax[0].legend()
    for graph in ax:
        graph.grid(), graph.set_xlabel('Metrics'), graph.set_ylabel('Performance')

    plt.sca(ax[0])
    plt.xticks(range(3), ['Micro F1', 'Macro F1', 'Accuracy'])
    plt.yticks(np.arange(0.4, 1.01, 0.05), np.arange(0.4, 1.01, 0.05))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.sca(ax[1])
    plt.xticks(range(3), ['Precision', 'Recall', 'F1 Score'])
    plt.yticks(np.arange(0.4, 1.01, 0.05), np.arange(0.4, 1.01, 0.05))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.sca(ax[2])
    plt.xticks(range(2), ['Class 0', 'Class 1'])

    plt.tight_layout()

    if oversample:
        ax[0].set_title('Model Performance - Oversampled')
        save_fig('classification_performance_oversampled')
    if not oversample:
        ax[0].set_title('Model Performance')
        save_fig('classification_performance')

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-50")
    plt.show()


def score_distribution():
    f, ax = plt.subplots(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    dtypes = {'Modelo': str, 'Prov': str, 'Local da Venda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}
    cols_to_use = ['Unnamed: 0', 'Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'stock_days', 'Margem', 'margem_percentagem']

    df = pd.read_csv('output/' + 'db_baviera_stock_optimization.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    # df = df[df['stock_days'] < 500]

    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    print(df['stock_days'].mean(), df['stock_days'].quantile(0.25), df['stock_days'].quantile(0.50), df['stock_days'].quantile(0.75), df['stock_days'].max())
    print(df[df['stock_days'] > 500].shape)

    # plt.plot(df['margem_percentagem'].sort_values(), df['margem_percentagem_norm'].sort_values())
    # plt.plot(df['stock_days'].sort_values(), df['inv_stock_days_norm'].sort_values())

    plt.plot(df['margem_percentagem'], df['stock_days'], 'o')
    p_corr = pearsonr(df['margem_percentagem'], df['stock_days'])

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    plt.grid()
    plt.title('Margin x Days in Stock - P.Corr= %.2f' % p_corr[0])
    plt.xlabel('Margin (%)')
    plt.ylabel('Days in Stock')
    save_fig('margin_vs_stock_days')
    plt.show()


def performance_evaluation_plot(input_dir):
    print('Performance Evaluation Plot')
    target_1 = 'score_class'
    target_2 = 'stock_class1'
    target_3 = 'stock_class2'
    target_4 = 'margem_class1'

    # none = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '.csv', index_col=0)
    # cols_grouped = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_group_cols.csv', index_col=0)
    # cols_grouped_prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_group_cols_prov_and_type.csv', index_col=0)
    # prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_prov_and_type.csv', index_col=0)
    none = pd.read_csv(input_dir + 'class_report_target_' + str(target_1) + '.csv', index_col=0)
    cols_grouped = pd.read_csv(input_dir + 'class_report_target_' + str(target_2) + '.csv', index_col=0)
    cols_grouped_prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target_3) + '.csv', index_col=0)
    prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target_4) + '.csv', index_col=0)

    # acc_none = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '.csv')
    # acc_cols_grouped = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_group_cols.csv')
    # acc_cols_grouped_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_group_cols_prov_and_type.csv')
    # acc_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_prov_and_type.csv')

    acc_none = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target_1) + '.csv')
    acc_cols_grouped = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target_2) + '.csv')
    acc_cols_grouped_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target_3) + '.csv')
    acc_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target_4) + '.csv')

    class_reports = [none, cols_grouped, prov_and_type, cols_grouped_prov_and_type]
    # class_reports = [none]

    acc = [acc_none, acc_cols_grouped, acc_prov_and_type, acc_cols_grouped_prov_and_type]
    # acc = [acc_none]

    precision, recall, f1_score, accuracy, micro, macro, i = [], [], [], [], [], [], 0
    for group in class_reports:
        precision.append(group.loc['avg/total', 'precision'])
        recall.append(group.loc['avg/total', 'recall'])
        f1_score.append(group.loc['avg/total', 'f1-score'])
        micro.append(acc[i]['micro_f1'])
        macro.append(acc[i]['macro_f1'])
        accuracy.append(acc[i]['accuracy'])
        i += 1

    plt.subplots(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    plt.plot(precision, label='Precision')
    plt.plot(recall, label='Recall')
    plt.plot(f1_score, label='F1 Score')
    plt.plot(accuracy, label='Accuracy')
    plt.xticks(range(0, 4), ['Score Class', 'stock_class1', 'stock_class2', 'margem_class1'])
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.xlabel('Approach Type')
    plt.ylabel('Performance')
    plt.title('Classification Performance vs Target Class')
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    save_fig('performance_evaluation_classification')
    plt.show()


def performance_evaluation_plot_all(input_dir, targets):

    f, ax = plt.subplots(2, 2, figsize=(1800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    pos = [list(i) for i in itertools.product([0, 1], repeat=2)]
    for target in targets:
        none = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '.csv', index_col=0)
        # cols_grouped = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_group_cols.csv', index_col=0)
        # cols_grouped_prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_group_cols_prov_and_type.csv', index_col=0)
        # prov_and_type = pd.read_csv(input_dir + 'class_report_target_' + str(target) + '_prov_and_type.csv', index_col=0)

        acc_none = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '.csv')
        # acc_cols_grouped = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_group_cols.csv')
        # acc_cols_grouped_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_group_cols_prov_and_type.csv')
        # acc_prov_and_type = pd.read_csv(input_dir + 'performance_evaluation_target_' + str(target) + '_prov_and_type.csv')

        class_reports = [none, cols_grouped, prov_and_type, cols_grouped_prov_and_type]
        acc = [acc_none, acc_cols_grouped, acc_prov_and_type, acc_cols_grouped_prov_and_type]

        precision, recall, f1_score, accuracy, micro, macro, i = [], [], [], [], [], [], 0
        for group in class_reports:
            precision.append(group.loc['avg/total', 'precision'])
            recall.append(group.loc['avg/total', 'recall'])
            f1_score.append(group.loc['avg/total', 'f1-score'])
            micro.append(acc[i]['micro_f1'])
            macro.append(acc[i]['macro_f1'])
            accuracy.append(acc[i]['accuracy'])
            i += 1

        ax[pos[targets.index(target)][0], pos[targets.index(target)][1]].plot(precision, label='Precision')
        ax[pos[targets.index(target)][0], pos[targets.index(target)][1]].plot(recall, label='Recall')
        ax[pos[targets.index(target)][0], pos[targets.index(target)][1]].plot(f1_score, label='F1 Score')
        ax[pos[targets.index(target)][0], pos[targets.index(target)][1]].plot(accuracy, label='Accuracy')
        plt.sca(ax[pos[targets.index(target)][0], pos[targets.index(target)][1]])
        plt.xticks(range(0, 4), ['Standard', 'Grouping Columns', 'w/Prov and Tipo Enc', 'Grouping Cols and w/Prov and Tipo Enc'])
        plt.title('Target - ' + str(target))

    for row, col in ax:
        row.legend(), col.legend()
        row.grid(), col.grid()
        row.set_xlabel('Approach Type'), col.set_xlabel('Approach Type')
        row.set_ylabel('Performance'), col.set_ylabel('Performance')

    plt.tight_layout()
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    save_fig('performance_evaluation_target_all_targets')
    plt.show()


def average_score(input_dir):
    targets = []

    df = pd.read_csv(input_dir + 'db_baviera_stock_optimization.csv', encoding='utf-8', delimiter=',', usecols=['Prov', 'Tipo Encomenda', 'stock_days', 'margem_percentagem'])
    df, _ = db_score_calculation(df, targets)

    # Prov
    column_grouping(df, column='Prov', values_to_keep=['Novos', 'Demonstração'])

    # Tipo Encomenda
    column_grouping(df, column='Tipo Encomenda', values_to_keep=['Enc Client Final', 'Encomenda Stock'])
    # print(df.head())

    print('Score:')
    for value in df['Prov_new'].unique():
        print(value, df.loc[df['Prov_new'] == value, 'score'].mean())

    print()

    print('Stock Days:')
    for value in df['Prov_new'].unique():
        print(value, df.loc[df['Prov_new'] == value, 'stock_days'].mean())

    print()

    print('Margem %:')
    for value in df['Prov_new'].unique():
        print(value, df.loc[df['Prov_new'] == value, 'margem_percentagem'].mean())

    print()

    print('Score:')
    for value in df['Tipo Encomenda_new'].unique():
        print(value, df.loc[df['Tipo Encomenda_new'] == value, 'score'].mean())

    print()

    print('Stock Days:')
    for value in df['Tipo Encomenda_new'].unique():
        print(value, df.loc[df['Tipo Encomenda_new'] == value, 'stock_days'].mean())

    print()

    print('Margem %:')
    for value in df['Tipo Encomenda_new'].unique():
        print(value, df.loc[df['Tipo Encomenda_new'] == value, 'margem_percentagem'].mean())

























if __name__ == '__main__':
    main()
