import pandas as pd
import numpy as np
import os
import glob
import csv
import itertools
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import roc_curve, auc, confusion_matrix
from gap_statistic import OptimalK
from xlsxwriter.workbook import Workbook
my_dpi = 96

'''
    File name: db_tools.py
    Author: Márcio Coelho
    Date created: 30/05/2018
    Date last modified: 27/06/2018
    Python Version: 3.6
'''


def null_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: '#null:'}))
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum() / df.shape[0] * 100).T.rename(index={0: '%null:'}))

    print(tab_info)


def zero_analysis(df):
    # Displays the number and percentage of null values in the DF

    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0)).T.rename(index={0: '#null:'}))
    tab_info = tab_info.append(pd.DataFrame((df == 0).astype(int).sum(axis=0) / df.shape[0] * 100).T.rename(index={0: '%null:'}))

    print(tab_info)


def save_csv(df, name):
    # Checks for file existence and deletes it if exists, then saves it

    if os.path.isfile(name):
        os.remove(name)
    df.to_csv(name)


def save_fig(name, save_dir='output/'):
    # Saves plot in at least two formats, png and pdf
    plt.savefig(save_dir + str(name) + '.pdf')
    plt.savefig(save_dir + str(name) + '.png')


def value_count_histogram(df, column, name, output_dir='output/'):
    plt.subplots(figsize=(1000 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    counts = df[column].value_counts().values
    values = df[column].value_counts().index
    rects = plt.bar(values, counts)

    # plt.tight_layout()
    plt.xlabel('Values')
    plt.xticks(rotation=30)
    plt.ylabel('Counts')
    plt.title('Distribution for column - ' + column)
    bar_plot_auto_label(rects)
    save_fig(name, output_dir)
    # plt.show()


def plot_roc_curve(models, models_name, train_x, train_y, test_x, test_y, save_name):
    # models_name = ['DT', 'RF', 'LR', 'SVM', 'AB', 'GC', 'Bayes', 'Voting']
    plt.subplots(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    for model in models:
        prob_train_init = model.fit(train_x, train_y).predict_proba(test_x)
        prob_test_1 = [x[1] for x in prob_train_init]
        fpr, tpr, _ = roc_curve(test_y, prob_test_1, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc + ' ' + str(models_name[models.index(model)]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic per Model')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    save_fig(save_name)
    # plt.show()
    plt.clf()
    plt.close()


def ohe(df, cols):

    for column in cols:
        uniques = df[column].unique()
        for value in uniques:
            new_column = column + '_' + str(value)
            df[new_column] = 0
            df.loc[df[column] == str(value), new_column] = 1
        df.drop(column, axis=1, inplace=True)

    return df


def bar_plot_auto_label(rects):

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')


def const_col_removal(df):

    list_before = list(df)
    for column in list_before:
        if df[column].nunique() == 1:
            df.drop(column, axis=1, inplace=True)
    list_after = list(df)
    print('Constant-columns removal:', [x for x in list_before if x not in list_after], '\n')

    return df


def df_standardization(df):

    scaler = StandardScaler()
    scaler.fit(df)
    scaled_matrix = scaler.transform(df)

    return scaled_matrix


def plot_confusion_matrix(test_y, best_model, prediction_test, tag, normalization=0, output_dir='output/'):

    # tn, fp, fn, tp = confusion_matrix(test_y, prediction_test).ravel()
    cm = confusion_matrix(test_y, prediction_test)
    tn, fp, fn, tp = cm.ravel()
    classes = best_model.classes_

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalization:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if normalization:
        save_fig('confusion_matrix_normalized_' + str(tag), save_dir=output_dir)
    elif not normalization:
        save_fig('confusion_matrix_' + str(tag), save_dir=output_dir)

    # plt.show()
    plt.clf()
    plt.close()

    return tn, fp, fn, tp


def plot_correlation_matrix(x, y, name, output_dir='output/'):
    x['Y'] = y

    plt.figure(figsize=(2000 / 96, 2000 / 96), dpi=96)
    plt.matshow(x.corr())
    plt.xticks(range(len(x.columns)), x.columns)
    plt.yticks(range(len(x.columns)), x.columns)
    plt.colorbar()
    plt.tight_layout()
    save_fig('correlation_matrix' + str(name), save_dir=output_dir)
    # plt.show()


def feature_selection(df, number_features):
    print('Feature Selection')

    selector = SelectKBest(mutual_info_regression, k=number_features).fit()


def gap_optimalk(matrix):

    optimalk = OptimalK(parallel_backend='joblib')
    k = optimalk(matrix, cluster_array=np.arange(1, 20))
    print('\nOptimal number of clusters is ', k)

    return k


def csv_to_excel_converter(file):
    for csvfile in glob.glob(os.path.join('.', file)):
        workbook = Workbook(csvfile[:-4] + '.xlsx')
        worksheet = workbook.add_worksheet()
        with open(csvfile, 'rt', encoding='utf8') as f:
            reader = csv.reader(f)
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet.write(r, c, col)
        workbook.close()


def unique_chassis_comparison(pse_sales, cm_bmw_mini):
    unique_chassis = pse_sales[pse_sales['nlr_code'] == '701']['chassis_number'].unique()
    print('Number of unique BMW/Mini Cars on PSE_Sales:', pse_sales[pse_sales['nlr_code'] == '701']['chassis_number'].nunique())
    print('Number of common chassis_numbers between PSE_Sales and CM BMW/MINI:', cm_bmw_mini[cm_bmw_mini['chassis_number'].isin(unique_chassis)].shape[0])


def graph_component_silhouette(X, n_clusters, lim_x, mat_size, sample_silhouette_values, silhouette_avg, clusters, method, approach, cluster_labels=0, cluster_centers=0):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    matplotlib.rc('patch', edgecolor='dimgray', linewidth=1)

    if cluster_labels.any():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
    elif not cluster_labels:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.8)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color='red', fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))

        ax1.axvline(x=silhouette_avg, color='white', ls='--', lw=1.0)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster label")

    if cluster_labels.any():
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = cluster_centers
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

        # ax2.set_title("The visualization of the clustered data.", fontsize=10)
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ""with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

    plt.tight_layout()
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    save_fig('stock_optimization_' + str(approach) + '_' + str(n_clusters) + '_cluster')
    plt.show()
