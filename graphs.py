import pandas as pd
import numpy as np
import itertools
import calendar
import scipy
import sys
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
    target = 'new_score'
    # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']
    targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class']
    oversample = 0
    score = 'recall'
    # score = 'f1_weighted'

    # performance_evaluation_plot(input_dir)  # 1
    # performance_evaluation_plot_all(input_dir, targets)  # 2
    # average_score(input_dir)  # 3
    # score_distribution()  # 4
    # score_distribution_alt()  # 4a
    # classification_report_plot(input_dir, oversample, score, target)  # 5

    # feature selection metrics: ['chi2', 'f_classif', 'mutual_info_classif']
    # models: ['dt', 'rf', 'lr', 'knn', 'svm', 'ab', 'gc', 'voting']
    criterium = ['chi2', 'f_classif', 'mutual_info_classif']
    # for feat_selection_metric in criterium:
    number_features_min, number_features_max, model, feat_selection_metric = 5, 33, 'dt', 'mutual_info_classif'
    # feature_selection_evaluation(input_dir, number_features_min, number_features_max, feat_selection_metric, model)  # 6

    sold_cars_evolution(input_dir)  # 7


# 7
def sold_cars_evolution(input_dir):
    print('7 - Evolution of cars sold per month per model')

    df = pd.read_csv(input_dir + 'db_baviera_grouped2.csv', index_col=0).dropna()
    total_cars_sold_per_month = [0] * 42
    # print(df.head())
    # sys.exit()

    plt.subplots(figsize=(1500 / my_dpi, 900 / my_dpi), dpi=my_dpi)
    models_list = list(df['Modelo_new'].unique())
    # models_list.remove('Outros')
    for model in models_list:
        number_cars_sold = []
        for year in [2015, 2016, 2017, 2018]:
            for month in range(1, 13):
                df_month = df[(df['Modelo_new'] == model) & (df['sell_year'] == year) & (df['sell_month'] == month)]
                if year == 2018 and month > 6:
                    continue

                number_cars_sold.append(df_month.shape[0])
        total_cars_sold_per_month = [total_cars_sold_per_month + cars_sold for (total_cars_sold_per_month, cars_sold) in zip(total_cars_sold_per_month, number_cars_sold)]
        print(model, number_cars_sold)
        plt.plot(number_cars_sold, '-o', label=model)

    all_pairs = []
    for year in [15, 16, 17, 18]:
        i = 0
        for month in ['Jan', 'Fev', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            i += 1
            if year == 18 and i > 6:
                continue
            pair = [month, year]
            all_pairs.append(pair[0])

    plt.plot(total_cars_sold_per_month, '-o', label='Month Total')
    plt.grid()
    plt.legend()
    plt.vlines(x=[11.5, 23.5, 35.5], ymin=0, ymax=300, color='red')
    plt.ylabel('Number of Sold Cars')
    plt.xlabel('Month, Year')
    plt.title('Number of Sold Cars per model, per month')
    plt.tight_layout()
    plt.xticks(range(0, 42), all_pairs, rotation=30)
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-10")
    save_fig('7_sold_cars_per_model_per_month_with_total')
    # print(total_cars_sold_per_month)
    plt.show()


# 6
def feature_selection_evaluation(input_dir, features_min, features_max, feat_sel_metric, model):
    print('Feature Selection Performance Evaluation')

    f, ax = plt.subplots(2, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    recall_class_0, recall_class_1, accuracy, specificity = [], [], [], []
    for value in range(features_min, features_max+1):
        cr = pd.read_csv(input_dir + 'class_report_classification_' + model + '_target_score_class_scoring_recall_oversample_' + str(value) + '_features_' + str(feat_sel_metric) + '.csv', index_col=0)
        pe = pd.read_csv(input_dir + 'performance_evaluation_classification_' + model + '_target_score_class_scoring_recall_oversample_' + str(value) + '_features_' + str(feat_sel_metric) + '.csv')
        recall_class_0.append(cr.loc['0.0', 'recall']), recall_class_1.append(cr.loc['1.0', 'recall'])
        accuracy.append(pe.loc[0, 'accuracy']), specificity.append(pe.loc[0, 'specificity'])

    ax[0].plot(recall_class_0, '-o', label='Recall Class 0'), ax[0].plot(recall_class_1, '-o', label='Recall Class 1')
    ax[1].plot(accuracy, '-o', label='Accuracy'), ax[1].plot(specificity, '-o', label='Specificity')
    ax[0].legend(), ax[1].legend()

    i = 0
    ax[0].set_title('Feature Selection Evaluation - Criteria: ' + feat_sel_metric)
    for graph in ax:
        graph.grid(), graph.set_xlabel('Number of Selected Features'), graph.set_ylabel('Performance')
        plt.sca(ax[i])
        plt.xticks(range(features_max-features_min+1), range(features_min, features_max+1))
        plt.yticks(np.arange(0.0, 1.01, 0.05), np.arange(0.0, 1.01, 0.05))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        i += 1
        plt.xlim([-1, 29])

    plt.tight_layout()
    save_fig('6_feature_selection_evaluation_' + feat_sel_metric + '_mini_removed')
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-10")
    plt.show()


# 5
def classification_report_plot(input_dir, oversample, score, target):

    # models = ['dt', 'rf', 'lr', 'knn', 'svm', 'ab', 'gc', 'voting']
    models = ['dt', 'rf', 'lr', 'ab', 'gc', 'bayes', 'neural', 'voting']
    # models_name = ['Dec. Tree', 'Rand. Forest', 'Log Reg', 'KNN', 'SVM', 'Adaboost', 'Gradient', 'Voting']
    models_name = ['Dec. Tree', 'Rand. Forest', 'Log Reg', 'AdaBoost', 'Gradient', 'Bayes', 'ANN', 'Voting']
    criterium = ['chi2', 'f_classif', 'mutual_info_classif']
    # for criteria in criterium:
    f, ax = plt.subplots(3, figsize=(1400 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    for model in models:
        model_label = models_name[models.index(model)]
        if oversample:
            cr = pd.read_csv(input_dir + 'class_report_classification_' + str(model) + '_target_score_class_scoring_' + str(score) + '_oversample.csv', index_col=0)
            pe = pd.read_csv(input_dir + 'performance_evaluation_classification_' + str(model) + '_target_score_class_scoring_' + str(score) + '_oversample.csv')
            # cr = pd.read_csv(input_dir + 'class_report_classification_' + str(model) + '_target_score_class_scoring_' + str(score) + '_oversample_10_features_' + str(criteria) + '.csv', index_col=0)
            # pe = pd.read_csv(input_dir + 'performance_evaluation_classification_' + str(model) + '_target_score_class_scoring_' + str(score) + '_oversample_10_features_' + str(criteria) + '.csv')
        if not oversample:
            cr = pd.read_csv(input_dir + 'class_report_classification_' + str(model) + '_target_' + str(target) + '_scoring_' + str(score) + '.csv', index_col=0)
            pe = pd.read_csv(input_dir + 'performance_evaluation_classification_' + str(model) + '_target_' + str(target) + '_scoring_' + str(score) + '.csv')

        ax[0].plot(pe.loc[0, :][:-1].values)
        ax[1].plot(cr.loc['avg/total', :][:-1].values, label=model_label)
        ax[2].plot(cr.loc[:, 'recall'][:-1].values)

    ax[1].legend()
    for graph in ax:
        graph.grid(), graph.set_xlabel('Metrics'), graph.set_ylabel('Performance')

    plt.sca(ax[0])
    plt.xticks(range(4), ['Micro F1', 'Macro F1', 'Accuracy', 'Specificity'])
    plt.yticks(np.arange(0.0, 1.01, 0.05), np.arange(0.0, 1.01, 0.05))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.sca(ax[1])
    plt.xticks(range(3), ['Precision', 'Recall/Sensitivity', 'F1 Score'])
    plt.yticks(np.arange(0.0, 1.01, 0.05), np.arange(0.0, 1.01, 0.05))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.sca(ax[2])
    plt.xticks(range(2), ['Recall - Class 0', 'Recall - Class 1'])
    plt.yticks(np.arange(0.0, 1.01, 0.05), np.arange(0.0, 1.01, 0.05))
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.ylim([0.2, 1.0])

    plt.tight_layout()

    if oversample:
        ax[0].set_title('Model Performance - Oversampled')
        save_fig('5_classification_performance_target_' + str(target) + '_scoring_' + str(score) + '_oversampled')
        # ax[0].set_title('Model Performance - Oversampled - ' + str(criteria))
        # save_fig('5_classification_performance_' + str(score) + '_oversampled_' + str(criteria))
    if not oversample:
        ax[0].set_title('Model Performance')
        save_fig('5_classification_performance_target_' + str(target) + '_scoring_' + str(score))
        # ax[0].set_title('Model Performance - ' + str(criteria))
        # save_fig('5_classification_performance_' + str(score) + '_' + str(criteria))

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-10")
    plt.show()


# 4
def score_distribution():
    f, ax = plt.subplots(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    dtypes = {'Modelo': str, 'Prov': str, 'Local da Venda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}
    cols_to_use = ['Unnamed: 0', 'Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'stock_days', 'Margem', 'margem_percentagem']

    df = pd.read_csv('output/' + 'db_baviera_stock_optimization.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    df = pd.read_csv('output/' + 'full_testing.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    # df = df[df['stock_days'] < 500]

    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    df['stock_days_norm_2'] = (max(df['stock_days']) * df['stock_days'] + 1) / (df['stock_days'] + 1)
    df['stock_days_norm_3'] = (df['stock_days_norm_2'] - df['stock_days_norm_2'].min()) / (df['stock_days_norm_2'].max() - df['stock_days_norm_2'].min())
    cdfy = scipy.stats.expon.cdf(df['stock_days'])
    df['stock_days_norm_4'] = scipy.stats.norm.ppf(cdfy)

    print(df.head(10))
    print()
    print(df.tail(10))

    # print(df['stock_days'].mean(), df['stock_days'].quantile(0.25), df['stock_days'].quantile(0.50), df['stock_days'].quantile(0.75), df['stock_days'].max())
    # print(df[df['stock_days'] > 500].shape)

    # plt.plot(range(df['stock_days'].shape[0]), df['stock_days'].sort_values(), 'o')
    # plt.plot(range(df['stock_days'].shape[0]), df['stock_days_norm_4'].sort_values(), '-')
    # plt.plot(range(df['stock_days'].shape[0]), df['stock_days_norm'].sort_values(), '-')
    # print(max(df['stock_days']))
    # print(range(0, 1001, 100))

    # hist, bin_edges = np.histogram(df['stock_days_norm'], bins=np.arange(0, 1, 0.01))
    # print(hist, '\n', bin_edges[:-1])
    # plt.bar(bin_edges[:-1], hist, width=0.005)

    # hist, bin_edges = np.histogram(df['margem_percentagem'], bins=range(-5, 8, 1))
    # print(hist, bin_edges)
    # plt.bar(bin_edges[:-1], hist)

    # plt.plot(range(df['margem_percentagem'].shape[0]), df['margem_percentagem'].sort_values())
    # plt.plot(df['stock_days'].sort_values(), df['inv_stock_days_norm'].sort_values())

    plt.plot(df['margem_percentagem'], df['stock_days'], 'o')
    p_corr = pearsonr(df['margem_percentagem'], df['stock_days'])

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    plt.grid()
    plt.title('Margin x Days in Stock - P.Corr= %.2f' % p_corr[0])
    plt.xlabel('Margin (%)')
    plt.ylabel('Days in Stock')
    # save_fig('4_margin_vs_stock_days')
    plt.show()


# 4a
def score_distribution_alt():
    stock_days = 1
    margin = 0
    # f, ax = plt.subplots(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    dtypes = {'Modelo': str, 'Prov': str, 'Local da Venda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float}
    cols_to_use = ['Unnamed: 0', 'Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'stock_days', 'Margem', 'margem_percentagem']

    # df = pd.read_csv('output/' + 'db_baviera_stock_optimization.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    df = pd.read_csv('output/' + 'db_full_baviera.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)

    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    # plt.plot(df['stock_days'], range(df.shape[0]), 'o')

    # hist, bin_edges = np.histogram(df['stock_days'], bins=np.arange(0, 1000, 100))
    # print(hist, '\n', bin_edges[:-1])
    # plt.bar(bin_edges[:-1], hist, width=5)

    if stock_days:
        # n, bin_edge = np.histogram(df['stock_days'], bins=np.arange(0, 1000, 10))
        # bincenters = 0.5 * (bin_edge[1:] + bin_edge[:-1])
        fig = plt.figure(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        new_tick_locations = np.array([df['stock_days'].min(), 10, 30, 45, 60, 90, 120, 150, 180, 200, 300, 400, 500, 600, 700, 800, 900, df['stock_days'].max()])
        # ax1.plot(bincenters, n, '-')
        ax1.plot(df['stock_days'], range(df.shape[0]), 'o')
        ax1.set_xlabel('Stock Days')
        ax1.set_xticks(new_tick_locations)
        ax1.set_xticklabels(new_tick_locations, rotation=30)

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations, stock_days, margin), rotation=30)
        ax2.set_xlabel('Stock Days Normalized')

        ax1.set_ylabel('Absolute Frequency')
        plt.axvline(x=45, color='red')
        wm = plt.get_current_fig_manager()
        wm.window.wm_geometry("-1500-100")
        plt.grid()
        plt.tight_layout()
        save_fig('4a_score_distribution_stock_days')
        plt.show()

    if margin:
        fig = plt.figure(figsize=(1400 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        # new_tick_locations = np.array([df['margem_percentagem'].min(), -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, df['margem_percentagem'].max()])
        new_tick_locations = np.array([df['margem_percentagem'].min(), -10, -5, 0, 3.5, 5, 10, df['margem_percentagem'].max()])

        # ax1.plot(bincenters, n, '-')
        ax1.plot(df['margem_percentagem'], range(df.shape[0]), 'o')
        ax1.set_xlabel('Margin')
        ax1.set_xticks(new_tick_locations)
        ax1.set_xticklabels(new_tick_locations, rotation=30)

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations, stock_days, margin), rotation=30)
        ax2.set_xlabel('Margin Normalized')

        ax1.set_ylabel('Absolute Frequency')
        plt.axvline(x=3.5, color='red')
        wm = plt.get_current_fig_manager()
        wm.window.wm_geometry("-1500-100")
        plt.grid()
        plt.tight_layout()
        save_fig('4a_score_distribution_margin')
        plt.show()


def tick_function(X, stock_days, margin):
    if stock_days:
        v = 1 - ((X - X.min()) / (X.max() - X.min()))
    if margin:
        v = (X - X.min()) / (X.max() - X.min())
    return ['%.4f' % z for z in v]


# 1
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


# 2
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
    save_fig('2_performance_evaluation_target_all_targets')
    plt.show()


# 3
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

