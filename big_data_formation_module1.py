import pandas as pd
import numpy as np
import sys
import csv
import os
import re
import time
import pickle
import pydotplus
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree, linear_model, ensemble, svm, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, SpectralClustering, Birch
from io import StringIO
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from db_tools import value_count_histogram, graph_component_silhouette, ohe, null_analysis, save_csv, plot_roc_curve, plot_confusion_matrix, plot_correlation_matrix
from classes import ClassFit, ClusterFit, RegFit
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/bin/'
pd.set_option('display.expand_frame_repr', False)
my_dpi = 96

'''
    File name: big_data_formation_module1.py
    Author: Márcio Coelho
    Python Version: 3.6
'''


def main():
    start = time.time()

    classification = 1
    sales_place_models = 0
    clustering = 0
    regression = 0

    retrain_models_check = 1  # If 1, retrains models. If 0, tries to use previously saved models

    # GridSearchCV Scoring Function:
    # score = 'accuracy'
    # score = 'roc_auc'
    score = 'recall'
    # score = 'f1_weighted'

    # Target to classify:
    target = ['new_score']
    if regression:
        target = ['margem_percentagem']
    # possible targets = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class', 'new_score']
    oversample = 0

    k = 10  # Number of Folds in Stratified Cross-Validation
    feat_sel_check = None  # Select whether to use Feature Selection. If None, will not use, if int, uses FS and int represents the number of features to select;
    # Feature Selection Criteria:
    # feature_selection_criteria = chi2
    # feature_selection_criteria = f_classif
    feature_selection_criteria = mutual_info_classif

    # Available Models:
    # models = ['dt']
    # models = ['rf']
    # models = ['lr']
    # models = ['knn']
    # models = ['svm']
    # models = ['ab']
    # models = ['gc']
    # models = ['bayes']
    # models = ['bagging']
    models = ['voting']
    # models = ['neural']
    # models = ['dt', 'rf', 'lr', 'knn', 'svm', 'ab', 'gc', 'bayes', 'voting']
    # models = ['dt', 'rf', 'lr', 'ab', 'gc', 'bayes', 'neural', 'voting']

    # for feat_sel_check in range(5, 34):
    #     print('Number of current Features Selected:', feat_sel_check)
    for model in models:
        name = tag(target, oversample, 'classification', score, feat_sel_check, feature_selection_criteria, model)

        df, train_x, train_y, test_x, test_y, ohe_cols, non_targets = database_preparation(oversample, target, feat_sel_check, feature_selection_criteria)
        # plot_correlation_matrix(train_x, train_y)
        # print(train_x.head(20))
        # sys.exit()

        if classification:
            if sales_place_models:
                stock_optimization_sales_place(df, target, start, oversample)  # Models by selling location - shouldn't be needed as the sales place should appear high in the tree;
            if not sales_place_models:
                stock_optimization_classification(df, name, model, k, score, train_x, train_y, test_x, test_y, ohe_cols, start, oversample, feat_sel_check, retrain_models_check)  # Classification Approach
        if clustering:
            stock_optimization_clustering(train_x, train_y, test_x, test_y, method='minibatchkmeans')  # Clustering Approach
        if regression:
            stock_optimization_regression(train_x, train_y, test_x, test_y, target)  # Regression Approach

        # customer_segmentation()

        print('\nRunning Time: %.2f' % (time.time() - start), 'seconds')


def database_preparation(oversample, target_column, feat_sel_check, feature_selection_criteria):
    print('Preparing database...')

    targets = ['margem_percentagem', 'Margem', 'stock_days']

    dtypes = {'Modelo': str, 'Prov': str, 'Local da Venda': str, 'Margem': float, 'Navegação': int, 'Sensores': int, 'Cor_Interior': str, 'Caixa Auto': int, 'Cor_Exterior': str, 'Jantes': str, 'stock_days': int, 'margem_percentagem': float, 'buy_day': str, 'buy_month': str, 'buy_year': str, 'price_total': float}
    cols_to_use = ['Unnamed: 0', 'Modelo', 'Prov', 'Local da Venda', 'Cor_Interior', 'Cor_Exterior', 'Navegação', 'Sensores', 'Caixa Auto', 'Jantes', 'stock_days', 'Margem', 'margem_percentagem', 'buy_day', 'buy_month', 'buy_year', 'sell_day', 'sell_month', 'sell_year', 'price_total']
    ohe_cols = ['Jantes_new', 'Cor_Interior_new', 'Cor_Exterior_new', 'Local da Venda_new', 'Modelo_new', 'Prov_new', 'buy_day', 'buy_month', 'buy_year']
    zscore_cols = ['price_total', 'number_prev_sales', 'last_margin', 'last_stock_days']

    df = pd.read_csv('output/' + 'db_full_baviera.csv', usecols=cols_to_use, encoding='utf-8', delimiter=',', index_col=0, dtype=dtypes)
    df = database_cleanup(df)
    df, targets = db_score_calculation(df, targets)

    col_group(df)

    # df.to_csv('output/df_baviera_grouped.csv')
    # sys.exit()

    targets += class_creation(df)
    df = new_features(df)

    z_scores_function(df, zscore_cols)

    df_ohe = df.copy(deep=True)
    df_ohe = ohe(df_ohe, ohe_cols)
    non_targets = [x for x in list(df_ohe) if x not in targets]

    if type(feat_sel_check) == int:
        df_ohe = feature_selection(df_ohe, non_targets, target_column, feat_sel_check, feature_selection_criteria)
    else:
        df_ohe = df_ohe[non_targets + target_column]

    df_train_x, df_train_y, df_test_x, df_test_y = dataset_split(df_ohe, target_column, oversample)

    return df, df_train_x, df_train_y, df_test_x, df_test_y, ohe_cols, non_targets


def new_features(df):
    sel_cols = ['Navegação', 'Sensores', 'Caixa Auto', 'Cor_Exterior_new', 'Cor_Interior_new', 'Jantes_new', 'Modelo_new']
    df['sell_date'] = pd.to_datetime(df['sell_year'] * 10000 + df['sell_month'] * 100 + df['sell_day'], format='%Y%m%d')
    df_grouped = df.sort_values(by=['sell_date']).groupby(sel_cols)
    df = df_grouped.apply(previous_sales_info)

    # df_grouped2 = df.sort_values(by=['sell_date']).groupby(sel_cols)
    # for key, group in df_grouped2:
    #     print(key, '\n', group)
    #
    # sys.exit()
    df.drop(['sell_date', 'sell_day', 'sell_month', 'sell_year'], axis=1, inplace=True)

    return df.fillna(0)
    # return df.dropna()


def previous_sales_info(x):

    prev_scores, i = [], 0
    if len(x) > 1:
        x['prev_sales_check'] = [0] + [1] * (len(x) - 1)
        x['number_prev_sales'] = list(range(len(x)))
        x['last_score'] = x['new_score'].shift(1)
        x['last_margin'] = x['margem_percentagem'].shift(1)
        x['last_stock_days'] = x['stock_days'].shift(1)

        for key, row in x.iterrows():
            prev_scores.append(x.loc[x.index == key, 'new_score'].values.tolist()[0])
            x.loc[x.index == key, 'average_score_dynamic'] = np.mean(prev_scores)
            if i == 0:
                x.loc[x.index == key, 'average_score_dynamic'] = 0
                i += 1

        x['average_score_dynamic_std'] = np.std(x['average_score_dynamic'])
        x['prev_average_score_dynamic'] = x['average_score_dynamic'].shift(1)  # New column - This one and following new column boost Adaboost for more than 15% in ROC Area
        x['prev_average_score_dynamic_std'] = x['average_score_dynamic_std'].shift(1)  # New column

        x['average_score_global'] = x['new_score'].mean()
        x['min_score_global'] = x['new_score'].min()
        x['max_score_global'] = x['new_score'].max()
        x['q3_score_global'] = x['new_score'].quantile(0.75)
        x['median_score_global'] = x['new_score'].median()
        x['q1_score_global'] = x['new_score'].quantile(0.25)

    elif len(x) == 0:
        x['prev_sales_check'] = 0
        x['number_prev_sales'] = 0
        x['last_score'] = 0
        # The reason I'm not filling all the other columns with zeros for the the len(x) == 0 case, is because I have a fillna(0) at the end of the function that called this one.

    return x


def z_scores_function(df, cols_to_normalize):
    for col in cols_to_normalize:
        df[col] = stats.zscore(df[col])

    return df


def feature_selection(df, features, target_column, feature_number, feature_selection_criteria):

    x = df[features]
    y = df[target_column]

    selector = SelectKBest(feature_selection_criteria, k=feature_number).fit(x, y)
    cols_selected = selector.get_support(indices=True)
    new_sel_features = list(list(list(x))[i] for i in cols_selected)
    new_sel_features += list(y)
    df_new = df[new_sel_features]

    return df_new


def stock_optimization_classification(df, name, model, k, score, train_x, train_y, test_x, test_y, ohe_cols, start, oversample, feat_sel_check, retrain_models_check):
    print('Classification Approach...')

    if oversample:
        oversample_flag_backup, original_index_backup = train_x['oversample_flag'], train_x['original_index']
        train_x.drop(['oversample_flag', 'original_index'], axis=1, inplace=True)

    # train_x['Y'] = train_y
    # print(train_x[['average_score_dynamic', 'average_score_dynamic_std', 'average_score_global', 'last_margin', 'last_score', 'last_stock_days', 'max_score_global', 'median_score_global', 'min_score_global', 'number_prev_sales', 'prev_average_score_dynamic', 'prev_average_score_dynamic_std', 'prev_sales_check', 'price_total', 'q1_score_global', 'q3_score_global', 'Y']].head(20))

    # sys.exit()

    if retrain_models_check:
        if model == 'dt':
            clf, clf_best = decision_tree(train_x, train_y, k, score, name)
        if model == 'rf':
            clf, clf_best = random_forest(train_x, train_y, k, score)
        if model == 'lr':
            clf, clf_best = logistic_regression(train_x, train_y, k, score)
        if model == 'knn':
            clf, clf_best = k_nearest_neighbours(train_x, train_y, k, score)
        if model == 'svm':
            clf, clf_best = support_vector_machine(train_x, train_y, k, score)
        if model == 'ab':
            clf, clf_best = adaboost_classifier(train_x, train_y, k, score)
        if model == 'gc':
            clf, clf_best = gradient_classifier(train_x, train_y, k, score)
        if model == 'bayes':
            clf, clf_best = bayesian_classifier(train_x, train_y, k, score)
        if model == 'bagging':
            clf, clf_best = bagging_classifier(train_x, train_y, k, score)
        if model == 'neural':
            clf, clf_best = neural_network(train_x, train_y, k, score)
        if model == 'voting':
            clf, clf_best = voting_method(train_x, train_y, test_x, test_y, k, score, name)

        save_model(clf, clf_best, model)
    elif not retrain_models_check:
        clf, clf_best = load_model(model)

    prediction_trainer, prediction_test = performance_evaluation(clf, clf_best, train_x, train_y, test_x, test_y, name, start)
    if model != 'voting':
        plot_roc_curve([clf_best], [model], train_x, train_y, test_x, test_y, 'roc_curve_' + name)
    if oversample:
        train_x['oversample_flag'], train_x['original_index'] = oversample_flag_backup, original_index_backup

    df_final = prediction_probabilities(clf_best, df, model, train_x, test_x, train_y, test_y, oversample, ohe_cols, prediction_trainer, prediction_test, feat_sel_check)
    df_new_data = new_columns(df_final)
    save_csv(df_new_data, 'output/db_final_' + str(name) + '.csv')


def new_columns(df):

    df['nr_cars_sold'] = 0
    cols_to_group = [x for x in list(df) if x in ['Caixa Auto', 'Navegação',  'Sensores', 'Jantes', 'Cor_Interior', 'Cor_Exterior', 'Local da Venda', 'Modelo', 'Prov']]
    df_grouped = df.groupby(cols_to_group)
    df = df_grouped.apply(additional_info)
    return df


def save_model(clf, clf_best, model):

    best_model_name = 'models/' + str(model) + '_best.sav'
    model_name = 'models/' + str(model) + '.sav'

    if os.path.isfile(model_name):
        os.remove(model_name)
        os.remove(best_model_name)

    pickle.dump(clf_best, open(best_model_name, 'wb'))
    pickle.dump(clf, open(model_name, 'wb'))


def load_model(model):

    best_model_name = 'models/' + str(model) + '_best.sav'
    model_name = 'models/' + str(model) + '.sav'

    clf = pickle.load(open(model_name, 'rb'))
    clf_best = pickle.load(open(best_model_name, 'rb'))

    return clf, clf_best


def additional_info(x):
    x['nr_cars_sold'] = len(x)
    x['average_percentage_margin'] = x['margem_percentagem'].mean()
    x['average_stock_days'] = int(x['stock_days'].mean())
    return x


def voting_method(train_x, train_y, test_x, test_y, k, score, name):
    print('Voting - Training models:')

    _, dt_best = decision_tree(train_x, train_y, k, score, name, voting=1)
    _, rf_best = random_forest(train_x, train_y, k, score, voting=1)
    _, lr_best = logistic_regression(train_x, train_y, k, score, voting=1)
    # _, svm_best = support_vector_machine(train_x, train_y, k, score, voting=1)
    # _, knn_best = k_nearest_neighbours(train_x, train_y, k, score, voting=1)
    _, bayes_best = bayesian_classifier(train_x, train_y, k, score, voting=1)
    _, ab_best = adaboost_classifier(train_x, train_y, k, score, voting=1)
    _, gc_best = gradient_classifier(train_x, train_y, k, score, voting=1)
    _, nn_best = neural_network(train_x, train_y, k, score, voting=1)

    print('### Voting ###')

    tuned_parameters_vote = {'voting': ['soft']}
    parameters = {'estimators': [('dt', dt_best), ('rf', rf_best), ('lr', lr_best), ('gc', gc_best), ('bayes', bayes_best), ('ab', ab_best), ('nn', nn_best)]}
    vote_clf = ClassFit(clf=ensemble.VotingClassifier, params=parameters)
    vote_clf.grid_search(parameters=tuned_parameters_vote, k=k, score=score)
    vote_clf.clf_fit(x=train_x, y=train_y)
    vote_clf_best = ensemble.VotingClassifier(estimators=[('dt', dt_best), ('rf', rf_best), ('lr', lr_best), ('gc', gc_best), ('bayes', bayes_best), ('ab', ab_best), ('nn', nn_best)], voting=vote_clf.grid.best_params_['voting'])
    vote_clf_best.fit(train_x, train_y)

    # models = [dt_best, rf_best, lr_best, svm_best, knn_best, ab_best, gc_best, bayes_best, nn_best, vote_clf_best]
    models = [dt_best, rf_best, lr_best, ab_best, gc_best, bayes_best, nn_best, vote_clf_best]
    # plot_roc_curve(models, ['Decision Tree', 'Random Forest', 'Log.Reg.', 'SVM', 'KNN', 'Adaboost', 'Gradient Class.', 'Bayesian',  'Neural Networks', 'Voting'], train_x, train_y, test_x, test_y, 'roc_curve_' + name)
    plot_roc_curve(models, ['Decision Tree', 'Random Forest', 'Log.Reg.', 'Adaboost', 'Gradient Class.', 'Bayesian',  'Neural Networks', 'Voting'], train_x, train_y, test_x, test_y, 'roc_curve_' + name)

    return vote_clf, vote_clf_best


def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def neural_network(train_x, train_y, k, score, voting=0):
    print('### Neural Networks ###')

    # v2
    tuned_parameters_nn = [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [1e-5], 'solver': ['sgd']}]
    nn = ClassFit(clf=MLPClassifier)
    nn.grid_search(parameters=tuned_parameters_nn, k=k, score=score)
    nn.clf_fit(x=train_x, y=train_y)

    nn_best = MLPClassifier(**nn.grid.best_params_)

    if not voting:
        nn_best.fit(train_x, train_y.values.ravel())

    return nn, nn_best


def decision_tree(train_x, train_y, k, score, name, voting=0):
    print('### Decision Tree ###')

    tuned_parameters_dt = [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 30], 'max_depth': [3, 5, 6], 'class_weight': ['balanced']}]
    dt = ClassFit(clf=tree.DecisionTreeClassifier)
    dt.grid_search(parameters=tuned_parameters_dt, k=k, score=score)
    dt.clf_fit(x=train_x, y=train_y)
    dt_best = tree.DecisionTreeClassifier(**dt.grid.best_params_)

    if not voting:
        dt_best.fit(train_x, train_y)
        feat_importance = dt_best.feature_importances_
        decision_tree_plot(dt_best, 'output/', name, train_x)
        feature_importance_graph(list(train_x), feat_importance, name)
        feature_importance_csv(list(train_x), feat_importance, name)

    return dt, dt_best


def random_forest(train_x, train_y, k, score, voting=0):
    print('### Random Forest ###')

    tuned_parameters_rf = [{'n_estimators': [10, 25, 50, 100], 'max_depth': [5, 10, 20], 'class_weight': ['balanced']}]
    rf = ClassFit(clf=RandomForestClassifier)
    rf.grid_search(parameters=tuned_parameters_rf, k=k, score=score)
    rf.clf_fit(x=train_x, y=train_y)

    rf_best = RandomForestClassifier(**rf.grid.best_params_)
    if not voting:
        rf_best.fit(train_x, train_y)

    return rf, rf_best


def logistic_regression(train_x, train_y, k, score, voting=0):
    print('### Logistic Regression ###')

    tuned_parameters_lr = [{'C': np.logspace(-2, 2, 20)}]
    lr = ClassFit(clf=linear_model.LogisticRegression)
    lr.grid_search(parameters=tuned_parameters_lr, k=k, score=score)
    lr.clf_fit(x=train_x, y=train_y.values.ravel())

    lr_best = linear_model.LogisticRegression(**lr.grid.best_params_)
    if not voting:
        lr_best.fit(train_x, train_y.values.ravel())

    return lr, lr_best


def k_nearest_neighbours(train_x, train_y, k, score, voting=0):
    print('### KNN ###')

    tuned_parameters_knn = [{'n_neighbors': np.arange(1, 50, 1)}]
    knn = ClassFit(clf=neighbors.KNeighborsClassifier)
    knn.grid_search(parameters=tuned_parameters_knn, k=k, score=score)
    knn.clf_fit(x=train_x, y=train_y)

    knn_best = neighbors.KNeighborsClassifier(**knn.grid.best_params_)
    if not voting:
        knn_best.fit(train_x, train_y)

    return knn, knn_best


def support_vector_machine(train_x, train_y, k, score, voting=0):
    print('### SVM ###')

    tuned_parameters_svc = [{'C': np.logspace(-2, 2, 10)}]
    # svc = ClassFit(clf=svm.LinearSVC)
    svc = ClassFit(clf=svm.SVC)  # Note: This performs better (around +5% on test dataset), but at the cost of at least x10 more time. Not worth it for now.
    svc.grid_search(parameters=tuned_parameters_svc, k=k, score=score)
    svc.clf_fit(x=train_x, y=train_y)

    # svc_best = svm.LinearSVC(**svc.grid.best_params_)
    svc_best = svm.SVC(**svc.grid.best_params_, probability=True)  # If using this model, there's a probability option for each class. Don't forget to adjust the code so this is saved in the final csv.
    if not voting:
        svc_best.fit(train_x, train_y)

    return svc, svc_best


def adaboost_classifier(train_x, train_y, k, score, voting=0):
    print('### Adaboost ###')

    tuned_parameters_ada = [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]
    ada = ClassFit(clf=AdaBoostClassifier)
    ada.grid_search(parameters=tuned_parameters_ada, k=k, score=score)
    ada.clf_fit(x=train_x, y=train_y)

    ada_best = AdaBoostClassifier(**ada.grid.best_params_)
    if not voting:
        ada_best.fit(train_x, train_y)

    return ada, ada_best


def gradient_classifier(train_x, train_y, k, score, voting=0):
    print('### Gradient ###')

    tuned_parameters_gb = [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]
    gb = ClassFit(clf=ensemble.GradientBoostingClassifier)
    gb.grid_search(parameters=tuned_parameters_gb, k=k, score=score)
    gb.clf_fit(x=train_x, y=train_y.values.ravel())

    gb_best = ensemble.GradientBoostingClassifier(**gb.grid.best_params_)
    if not voting:
        gb_best.fit(train_x, train_y.values.ravel())

    return gb, gb_best


def bayesian_classifier(train_x, train_y, k, score, voting=0):
    print('### Bayesian ###')

    gnb = ClassFit(clf=GaussianNB)
    gnb_best = GaussianNB().fit(train_x, train_y)

    return gnb, gnb_best


def bagging_classifier(train_x, train_y, k, score, voting=0):
    print('### Bagging ###')

    # Random Forest:
    # bag = ClassFit(clf=BaggingClassifier)
    # tuned_parameters_bag = {'base_estimator__n_estimators': [10, 25, 50, 100], 'base_estimator__max_depth': [5, 10, 20], 'base_estimator__class_weight': ['balanced'], 'max_samples': [0.005, 0.1, 0.2, 0.5]}
    # bag_2 = GridSearchCV(BaggingClassifier(RandomForestClassifier(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_bag, cv=k, scoring=score)
    # bag_2.fit(train_x, train_y)
    # bag_best = BaggingClassifier(RandomForestClassifier(n_estimators=bag_2.best_params_['base_estimator__n_estimators'], max_depth=bag_2.best_params_['base_estimator__max_depth'], class_weight=bag_2.best_params_['base_estimator__class_weight']))
    # bag_best.fit(train_x, train_y)

    # Bayesian Classifier:
    # bag = ClassFit(clf=GaussianNB)
    # tuned_parameters_bag = {'max_samples': [0.005, 0.1, 0.2, 0.5]}
    # bag_2 = GridSearchCV(BaggingClassifier(GaussianNB(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_bag, cv=k, scoring=score)
    # bag_2.fit(train_x, train_y)
    # bag_best = BaggingClassifier(GaussianNB())
    # bag_best.fit(train_x, train_y)

    # tuned_parameters_svc = [{'C': np.logspace(-2, 2, 10)}]
    # svc = ClassFit(clf=svm.LinearSVC)
    # svc.grid_search(parameters=tuned_parameters_svc, k=k, score=score)
    # svc.clf_fit(x=train_x, y=train_y)

    # SVM
    # bag = ClassFit(clf=svm.LinearSVC)
    # tuned_parameters_svm = {'base_estimator__C': np.logspace(-2, 2, 10), 'max_samples': [0.005, 0.1, 0.2, 0.5]}
    # bag_2 = GridSearchCV(BaggingClassifier(svm.LinearSVC(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_svm, cv=k, scoring=score)
    # bag_2.fit(train_x, train_y)
    # bag_best = BaggingClassifier(svm.LinearSVC(C=bag_2.best_params_['base_estimator__C']))
    # bag_best.fit(train_x, train_y)

    # LR
    # bag = ClassFit(clf=linear_model.LogisticRegression)
    # tuned_parameters_lr = {'base_estimator__C': np.logspace(-2, 2, 10), 'max_samples': [0.005, 0.1, 0.2, 0.5]}
    # bag_2 = GridSearchCV(BaggingClassifier(linear_model.LogisticRegression(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_lr, cv=k, scoring=score)
    # bag_2.fit(train_x, train_y)
    # bag_best = BaggingClassifier(linear_model.LogisticRegression(C=bag_2.best_params_['base_estimator__C']))
    # bag_best.fit(train_x, train_y)

    # GC
    # bag = ClassFit(clf=ensemble.GradientBoostingClassifier)
    # tuned_parameters_gc = {'base_estimator__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    # bag_2 = GridSearchCV(BaggingClassifier(ensemble.GradientBoostingClassifier(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_gc, cv=k, scoring=score)
    # bag_2.fit(train_x, train_y)
    # bag_best = BaggingClassifier(ensemble.GradientBoostingClassifier(n_estimators=bag_2.best_params_['base_estimator__n_estimators']))
    # bag_best.fit(train_x, train_y)

    # Adaboost
    bag = ClassFit(clf=AdaBoostClassifier)
    tuned_parameters_ada = {'base_estimator__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    bag_2 = GridSearchCV(BaggingClassifier(AdaBoostClassifier(), n_estimators=100, max_features=0.5), param_grid=tuned_parameters_ada, cv=k, scoring=score)
    bag_2.fit(train_x, train_y)
    bag_best = BaggingClassifier(AdaBoostClassifier(n_estimators=bag_2.best_params_['base_estimator__n_estimators']))
    bag_best.fit(train_x, train_y)

    return bag, bag_best


def prediction_probabilities(clf, df, model, train_x, test_x, train_y, test_y, oversample, ohe_cols, prediction_trainer, prediction_test, feat_sel_check):

    if oversample:
        oversample_flag_backup, original_index_backup = train_x['oversample_flag'], train_x['original_index']
        train_x.drop(['oversample_flag', 'original_index'], axis=1, inplace=True)

    if model != 'svm':
        prob_train_init = clf.predict_proba(train_x)
        prob_test_init = clf.predict_proba(test_x)
        prob_train_0 = [x[0] for x in prob_train_init]
        prob_train_1 = [x[1] for x in prob_train_init]
        prob_test_0 = [x[0] for x in prob_test_init]
        prob_test_1 = [x[1] for x in prob_test_init]
        train_x['proba_0'] = prob_train_0
        train_x['proba_1'] = prob_train_1

        train_x['score_class_gt'] = train_y
        train_x['score_class_pred'] = prediction_trainer

        if oversample:
            train_x['oversample_flag'], train_x.index = oversample_flag_backup, original_index_backup
            train_x.drop_duplicates(subset=['oversample_flag'], inplace=True)
            train_x.drop(['oversample_flag'], axis=1, inplace=True)

        # if not oversample:
        #     train_x = pd.concat([train_x, df['score']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['stock_days']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['Margem']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['margem_percentagem']], join='inner', axis=1)

        test_x['score_class_gt'] = test_y
        test_x['score_class_pred'] = prediction_test
        test_x['proba_0'] = prob_test_0
        test_x['proba_1'] = prob_test_1

        # if not oversample:
        #     test_x = pd.concat([test_x, df['score']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['stock_days']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['Margem']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['margem_percentagem']], join='inner', axis=1)

        # plot_roc(test_y, [prob_test_0, prob_test_1])

    if model == 'svm':
        probability_trainer = clf.decision_function(train_x).tolist()
        probability_test = clf.decision_function(test_x).tolist()
        train_x['decision_function'] = probability_trainer
        test_x['decision_function'] = probability_test

        train_x['score_class_gt'] = train_y
        train_x['score_class_pred'] = prediction_trainer

        if oversample:
            train_x['oversample_flag'], train_x.index = oversample_flag_backup, original_index_backup
            train_x.drop_duplicates(subset=['oversample_flag'], inplace=True)
            train_x.drop(['oversample_flag'], axis=1, inplace=True)

        # if not oversample:
        #     train_x = pd.concat([train_x, df['score']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['stock_days']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['Margem']], join='inner', axis=1)
        #     train_x = pd.concat([train_x, df['margem_percentagem']], join='inner', axis=1)

        test_x['score_class_gt'] = test_y
        test_x['score_class_pred'] = prediction_test
        # if not oversample:
        #     test_x = pd.concat([test_x, df['score']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['stock_days']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['Margem']], join='inner', axis=1)
        #     test_x = pd.concat([test_x, df['margem_percentagem']], join='inner', axis=1)

    train_x = pd.concat([train_x, df['score']], join='inner', axis=1)
    train_x = pd.concat([train_x, df['stock_days']], join='inner', axis=1)
    train_x = pd.concat([train_x, df['Margem']], join='inner', axis=1)
    train_x = pd.concat([train_x, df['margem_percentagem']], join='inner', axis=1)

    test_x = pd.concat([test_x, df['score']], join='inner', axis=1)
    test_x = pd.concat([test_x, df['stock_days']], join='inner', axis=1)
    test_x = pd.concat([test_x, df['Margem']], join='inner', axis=1)
    test_x = pd.concat([test_x, df['margem_percentagem']], join='inner', axis=1)

    df_new = pd.concat([train_x, test_x], axis=0, sort=True)
    df_new = reversed_ohe(df_new, ohe_cols)

    return df_new


def reversed_ohe(df, ohe_cols):
    for value in ohe_cols:
        ohe_columns = [x for x in list(df) if value in x]
        if len(ohe_columns):
            col_name = value.replace('_new', '')
            df[col_name] = pd.get_dummies(df[ohe_columns]).idxmax(1)
            df[col_name] = df[col_name].str.replace((value + '_'), '')
            df.drop(ohe_columns, axis=1, inplace=True)

    return df


def db_score_calculation(df, targets):
    df['stock_days_norm'] = (df['stock_days'] - df['stock_days'].min()) / (df['stock_days'].max() - df['stock_days'].min())
    df['inv_stock_days_norm'] = 1 - df['stock_days_norm']

    df['margem_percentagem_norm'] = (df['margem_percentagem'] - df['margem_percentagem'].min()) / (df['margem_percentagem'].max() - df['margem_percentagem'].min())
    df['score'] = df['inv_stock_days_norm'] * df['margem_percentagem_norm']

    df['stock_days_class'] = 0
    # df.loc[df['inv_stock_days_norm'] > 0.97, 'stock_days_class'] = 1
    df.loc[df['stock_days'] <= 45, 'stock_days_class'] = 1
    df['margin_class'] = 0
    # df.loc[df['margem_percentagem_norm'] > 0.60, 'margin_class'] = 1
    df.loc[df['margem_percentagem'] >= 3.5, 'margin_class'] = 1

    df.drop(['stock_days_norm', 'inv_stock_days_norm', 'margem_percentagem_norm'], axis=1, inplace=True)
    targets += ['score']

    return df, targets


def stock_optimization_clustering(train_x, train_y, test_x, test_y, method):
    print('Clustering Approach...')

    n_clusters = 10
    parameters = {'n_clusters': n_clusters, 'batch_size': 10000, 'n_init': 50}
    kmeans = ClusterFit(clf=MiniBatchKMeans, params=parameters)
    kmeans.clf_fit(x=train_x)
    cluster_clients = kmeans.predict(x=test_x)
    kmeans.silhouette_score_avg(x=test_x, cluster_clients=cluster_clients)
    kmeans.sample_silhouette_score(x=test_x, cluster_clients=cluster_clients)
    matrix = df_standardization(test_x)
    graph_component_silhouette(matrix, n_clusters, [-0.1, 1.0], len(matrix), kmeans.sample_silh_score, kmeans.silh_score, cluster_clients, method, 'clustering', kmeans.labels(), kmeans.cluster_centers())

    # n_clusters = 29
    # parameters = {'n_clusters': n_clusters, 'batch_size': 10000, 'n_init': 50}
    # kmeans = ClusterFit(clf=MiniBatchKMeans, params=parameters)
    # matrix_train = df_standardization(train_x)
    # matrix_test = df_standardization(test_x)
    # # kmeans.cluster_optimal_number(matrix_train)
    # kmeans.clf_fit(x=matrix_train)
    # cluster_clients = kmeans.predict(x=matrix_test)
    # kmeans.silhouette_score_avg(x=matrix_test, cluster_clients=cluster_clients)
    # kmeans.sample_silhouette_score(x=matrix_test, cluster_clients=cluster_clients)
    # graph_component_silhouette(matrix_test, n_clusters, [-0.1, 1.0], len(matrix_test), kmeans.sample_silh_score, kmeans.silh_score, cluster_clients, method, 'clustering', kmeans.labels(), kmeans.cluster_centers())


def stock_optimization_regression(train_x, train_y, test_x, test_y, target_column):
    print('Regression Approach...')

    tuned_parameters = {'fit_intercept': [True]}
    reg = RegFit(clf=linear_model.LinearRegression, params=tuned_parameters)
    # reg.grid_search(tuned_parameters, k=10, score='f1_weighted')
    reg.clf_fit(x=train_x, y=train_y)
    # reg_best = linear_model.LinearRegression(**reg.grid.best_params_)
    # reg_best.fit(train_x, train_y)

    name = tag(target_column, 'regression')
    prediction_test = reg.predict(test_x)
    prediction_train = reg.predict(train_x)
    coef = reg.coefficients()
    mse_test = reg.mse_func(prediction_test, test_y)
    score_test = reg.score_func(prediction_test, test_y)  # Explained variance score: 1 is perfect prediction and 0 means that there is no linear relationship between X and y.

    mse_train = reg.mse_func(prediction_train, train_y)
    score_train = reg.score_func(prediction_train, train_y)  # Explained variance score: 1 is perfect prediction and 0 means that there is no linear relationship between X and y.

    # print('Coefficients:', coef)
    print('Train - R^2: %.3f' % score_train)
    print('Train - Mean Square Error: %.3f' % mse_train)

    print('Test - R^2: %.3f' % score_test)
    print('Test - Mean Square Error: %.3f' % mse_test)


def df_standardization(df):

    scaler = StandardScaler()
    scaler.fit(df)
    scaled_matrix = scaler.transform(df)

    return scaled_matrix


def database_cleanup(df):

    df = df[~df.Modelo.str.contains('Série')]  # Removes all Motorcycles
    df = df[~df.Modelo.str.contains('Z4')]  # Removes Z4 Models
    df = df.loc[df['Prov'] != 'Demonstração']  # Removes demo cars
    df = df.loc[df['Prov'] != 'Em utilização']  # Removes cars being used
    df = df[~df.Modelo.str.contains('MINI')]  # Removes all Mini's

    return df


def stock_optimization_sales_place(df, target_column, start, oversample):
    print('Models per Sales Place')

    df_copy = df
    sale_places = [x for x in list(df) if 'Local da Venda' in x]
    for value in sale_places:
        print('###', value)
        df = df_copy.loc[df_copy[value] == 1]

        train_x, train_y, test_x, test_y = dataset_split(df, target_column, oversample)

        tuned_parameters_dt = [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 50], 'max_depth': [3, 5], 'class_weight': ['balanced']}]
        dt = ClassFit(clf=tree.DecisionTreeClassifier)
        dt.grid_search(parameters=tuned_parameters_dt, k=10, score='f1_weighted')
        dt.clf_fit(x=train_x, y=train_y)

        dt_best = tree.DecisionTreeClassifier(**dt.grid.best_params_)
        dt_best.fit(train_x, train_y)
        name = tag(target_column, 'classification', value)
        decision_tree_plot(dt_best, 'output/', name, train_x)

        feat_importance = dt_best.feature_importances_
        performance_evaluation(dt, dt_best, train_x, train_y, test_x, test_y, name, start)
        feature_importance_graph(list(train_x), feat_importance, name)


def customer_segmentation():
    df = pd.read_csv('sql_db/' + 'DataSet_Customer_Segmentation.csv', delimiter=';', index_col=0, header=None, dtype={3: str, 4: str})
    # renames = {1: 'cliente_fatura', 2: 'empresa', 3: 'data_fatura', 4: 'marca', 5: 'modelo', 6: 'km', 7: 'dt_entrada', 8:'dt_matricula', 9:'anos_viatura', 10: 'soldbygsc', 11:'tipo_venda_nivel_1', 12:'tipo_venda_nivel_1', 13:'departamento', 14:'valor_faturado', 15:'visita_em garantia', 16: 'cm_check', 21: 'visita_em_cm'}
    # cols_to_drop = [17, 18, 19, 20]
    # df.rename(renames, axis=1, inplace=True)
    # df.drop(cols_to_drop, axis=1, inplace=True)
    print(df.head())


def decision_tree_plot(clf, output_dir, tag_id, df_train_x, sales_place=0):
    print('Plotting Decision Tree...')
    file_name = 'decision_tree_' + str(tag_id)
    if sales_place:
        file_name += str(sales_place)

    # dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, feature_names=list(df_train_x), class_names=['0', '1', '2', '3'], special_characters=True)
    dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, feature_names=list(df_train_x), class_names=str(clf.classes_), special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(output_dir + file_name + '.pdf')


def target_cols_removal(df, target, targets_to_remove):

    targets_to_remove.remove(target[0])
    for value in targets_to_remove:
        df.drop(value, axis=1, inplace=True)

    return df


def oversample_data(train_x, train_y):

    train_x['oversample_flag'] = range(train_x.shape[0])
    train_x['original_index'] = train_x.index
    # print(train_x.shape, '\n', train_y['new_score'].value_counts())

    ros = RandomOverSampler(random_state=42)
    # ros = RandomUnderSampler(random_state=42)
    # sm = SMOTE(random_state=42)
    train_x_resampled, train_y_resampled = ros.fit_sample(train_x, train_y.values.ravel())
    # train_x_resampled, train_y_resampled = sm.fit_sample(train_x, train_y.values.ravel())

    train_x_resampled = pd.DataFrame(np.atleast_2d(train_x_resampled), columns=list(train_x))
    train_y_resampled = pd.Series(train_y_resampled)
    for column in list(train_x_resampled):
        if train_x_resampled[column].dtype != train_x[column].dtype:
            # print('Problem found with dtypes, fixing it...', )
            dtype_checkup(train_x_resampled, train_x)
        break

    # print(train_x_resampled.shape, '\n', train_y_resampled.value_counts())
    return train_x_resampled, train_y_resampled


def dtype_checkup(train_x_resampled, train_x):
    for column in list(train_x):
        train_x_resampled[column] = train_x_resampled[column].astype(train_x[column].dtype)


def col_group(df):
    # Cor_Exterior
    color_ext_grouping(df)
    # value_count_histogram(df, 'Cor_Exterior', 'cor_exterior_before')
    # value_count_histogram(df, 'Cor_Exterior_new', 'cor_exterior_after_mini_removal')

    # Cor_Interior
    color_int_grouping(df)
    # value_count_histogram(df, 'Cor_Interior', 'cor_interior_before')
    # value_count_histogram(df, 'Cor_Interior_new', 'cor_interior_after_mini_removal')

    # Jantes
    jantes_grouping(df)
    # value_count_histogram(df, 'Jantes', 'jantes_before')
    # value_count_histogram(df, 'Jantes_new', 'jantes_after_mini_removal')

    # Modelo
    model_grouping(df)
    # value_count_histogram(df, 'Modelo', 'modelo_before')
    # value_count_histogram(df, 'Modelo_new', 'modelo_after_mini_removal')

    # Local da Venda
    sales_place_grouping(df)
    # value_count_histogram(df, 'Local da Venda', 'local_da_venda_before')
    # value_count_histogram(df, 'Local da Venda_new', 'local_da_venda_after_mini_removal')

    # Prov
    # value_count_histogram(df, 'Prov', 'prov_before')
    df.loc[df['Prov'] == 'Viaturas Km 0', 'Prov'] = 'Novos'
    df.rename({'Prov': 'Prov_new'}, axis=1, inplace=True)
    # value_count_histogram(df, 'Prov_new', 'prov_after_mini_removal')

    df.drop('Cor_Exterior', axis=1, inplace=True)
    df.drop('Cor_Interior', axis=1, inplace=True)
    df.drop('Jantes', axis=1, inplace=True)
    df.drop('Modelo', axis=1, inplace=True)
    df.drop('Local da Venda', axis=1, inplace=True)


def color_ext_grouping(df):

    preto = ['preto']
    cinzento = ['cinzento', 'prateado', 'prata', 'cinza']
    branco = ['branco']
    azul = ['azul', 'bluestone']
    verde = ['verde']
    vermelho_laranja = ['vermelho', 'laranja']
    burgundy = ['burgundy']
    castanho = ['castanho']
    others = ['jatoba', 'aqua', 'storm', 'cedar', 'bronze', 'chestnut', 'havanna', 'cashmere', 'champagne', 'dourado', 'amarelo', 'bege', 'silverstone', 'moonstone']

    groups = [preto, cinzento, branco, azul, verde, vermelho_laranja, burgundy, castanho, others]
    groups_name = ['preto', 'cinzento', 'branco', 'azul', 'verde', 'vermelho/laranja', 'burgundy', 'castanho', 'outros']
    for group in groups:
        for dc in group:
            df.loc[df['Cor_Exterior'] == dc, 'Cor_Exterior_new'] = groups_name[groups.index(group)]

    return df


def color_int_grouping(df):

    preto = ['preto', 'prata/preto/preto', 'veneto/preto', 'preto/preto', 'ambar/preto/preto']
    antracite = ['antracite', 'antracite/cinza/preto', 'antracite/preto', 'antracite/vermelho/preto', 'antracite/vermelho', 'anthtacite/preto', 'anthracite/silver']
    castanho = ['dakota', 'castanho', 'oak', 'terra', 'mokka', 'vernasca']
    # cinzento = ['cinzento']
    # azul = ['azul']
    # bege = ['oyster','bege','oyster/preto']
    # branco = ['branco']
    others = ['champagne', 'branco', 'oyster', 'prata/cinza', 'bege', 'oyster/preto', 'azul', 'cinzento', 'truffle', 'burgundy', 'zagora/preto', 'sonoma/preto', 'laranja', 'taupe/preto', 'vermelho', 'silverstone', 'nevada', 'cognac/preto', 'preto/laranja', 'preto/prateado']

    groups = [preto, antracite, castanho, others]
    groups_name = ['preto', 'antracite', 'castanho', 'outros']
    for group in groups:
        for dc in group:
            df.loc[df['Cor_Interior'] == dc, 'Cor_Interior_new'] = groups_name[groups.index(group)]

    return df


def jantes_grouping(df):
    standard = ['standard', '15', '16']
    seventeen_pol = ['17']
    eighteen_pol = ['18']
    nineteen_or_twenty_pol = ['19', '20']

    groups = [standard, seventeen_pol, eighteen_pol, nineteen_or_twenty_pol]
    groups_name = ['Standard', '17', '18', '19/20']
    for group in groups:
        for dc in group:
            df.loc[df['Jantes'] == dc, 'Jantes_new'] = groups_name[groups.index(group)]

    return df


def sales_place_grouping(df):
    # 1st grouping:
    # norte_group = ['DCC - Feira', 'DCG - Gaia', 'DCV - Coimbrões', 'DCN-Porto', 'DCN-Porto Mini', 'DCC - Aveiro', 'DCG - Gaia Mini']
    # norte_group_used = ['DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
    # center_group = ['DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi']
    # center_group_used = ['DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad']
    # south_group = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro']
    # south_group_used = ['DCA -Portimão Usados']

    # 2nd grouping:
    centro = ['DCV - Coimbrões', 'DCC - Aveiro']
    norte = ['DCC - Feira', 'DCG - Gaia', 'DCN-Porto', 'DCN-Porto Mini', 'DCG - Gaia Mini', 'DCN-Porto Usados', 'DCG - Gaia Usados', 'DCC - Feira Usados', 'DCC - Aveiro Usados', 'DCC - Viseu Usados']
    sul = ['DCS-Expo Frotas Busi', 'DCS-V Especiais BMW', 'DCS-V Especiais MINI', 'DCS-Expo Frotas Flee', 'DCS-Cascais', 'DCS-Parque Nações', 'DCS-Parque Nações Mi', 'DCS-24 Jul BMW Usad', 'DCS-Cascais Usados', 'DCS-24 Jul MINI Usad', 'DCS-Lisboa Usados']
    algarve = ['DCA - Faro', 'DCA - Portimão', 'DCA - Mini Faro', 'DCA -Portimão Usados']
    motorcycles = ['DCA - Motos Faro', 'DCS- Vendas Motas', 'DCC - Motos Aveiro']

    # groups = [norte_group, norte_group_used, center_group, center_group_used, south_group, south_group_used, motorcycles, unknown_group]
    # groups_name = ['norte', 'norte_usados', 'centro', 'centro_usados', 'sul', 'sul_usados', 'motos', 'unknown']
    groups = [norte, centro, sul, motorcycles, algarve]
    groups_name = ['norte', 'centro', 'sul', 'motos', 'algarve']
    for group in groups:
        for dc in group:
            df.loc[df['Local da Venda'] == dc, 'Local da Venda_new'] = groups_name[groups.index(group)]

    # print(df[['Local da Venda', 'Local da Venda New']])
    return df


def model_grouping(df):

    # s1 = ['S1 3p', 'S1 5p']
    # s2 = ['S2 Active Tourer', 'S2 Cabrio', 'S2 Gran Tourer', 'S2 Coupé']
    # s3 = ['S3 Touring', 'S3 Gran Turismo', 'S3 Berlina']
    # s4 = ['S4 Gran Coupé', 'S4 Coupé']
    # s5 = ['S5 Touring', 'S5 Limousine', 'S5 Gran Turismo', 'S5 Berlina']
    # s6 = ['S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé']
    # s7 = ['S7 Berlina', 'S7 L Berlina']
    # x1 = ['X1']
    # x2 = ['X2 SAC']
    # x3 = ['X3 SUV']
    # x4 = ['X4 SUV']
    # x5 = ['X5 SUV', 'X5 M']
    # x6 = ['X6' 'X6 M']
    # z4 = ['Z4 Roadster']
    # mini = ['MINI 5p', 'MINI 3p', 'MINI CLUBMAN', 'MINI CABRIO', 'MINI COUNTRYMAN']

    s2_gran = ['S2 Gran Tourer']
    s2_active = ['S2 Active Tourer']
    s3_touring = ['S3 Touring']
    s3_berlina = ['S3 Berlina']
    s4_gran = ['S4 Gran Coupé']
    s5_touring = ['S5 Touring']
    s5_lim_ber = ['S5 Limousine', 'S5 Berlina']
    s1 = ['S1 3p', 'S1 5p']
    x1 = ['X1']
    x3 = ['X3 SUV']
    mini_club = ['MINI CLUBMAN']
    mini_cabrio = ['MINI CABRIO']
    mini_country = ['MINI COUNTRYMAN']
    mini = ['MINI 5p', 'MINI 3p']
    motos = ['Série C', 'Série F', 'Série K', 'Série R']
    outros = ['S2 Cabrio', 'S2 Gran Tourer', 'S2 Coupé', 'S3 Gran Turismo', 'S4 Coupé', 'S4 Cabrio', 'S5 Gran Turismo', 'S6 Cabrio', 'S6 Gran Turismo', 'S6 Gran Coupe', 'S6 Coupé', 'S7 Berlina', 'S7 L Berlina', 'X2 SAC', 'X4 SUV', 'X5 SUV', 'X5 M', 'X6', 'X6 M', 'Z4 Roadster', 'M2 Coupé', 'M3 Berlina', 'M4 Cabrio', 'M4 Coupé', 'S6 Gran Turismo', 'S6 Cabrio', 'S6 Coupé', 'S6 Gran Coupe', 'S7 Berlina', 'S7 L Berlina']

    # groups = [s1, s2, s3, s4, s5, s6, s7, x1, x2, x3, x4, x5, x6, z4, motos, mini]
    groups = [s1, s2_gran, s2_active, s3_touring, s3_berlina, s4_gran, s5_touring, s5_lim_ber, x1, x3, motos, mini, mini_club, mini_cabrio, mini_country, outros]
    # groups_name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Z4', 'Motociclos', 'Mini']
    groups_name = ['S1', 'S2_gran', 'S2_active', 'S3_touring', 'S3_berlina', 'S4_Gran', 'S5_Touring', 'S5_Lim_Ber', 'X1', 'X3', 'Motociclos', 'Mini', 'Mini_Club', 'Mini_Cabrio', 'Mini_Country', 'Outros']
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
    # df.loc[(df['score'] >= 0) & (df['score'] < 0.25), 'score_class'] = 0
    # df.loc[(df['score'] >= 0.25) & (df['score'] < 0.50), 'score_class'] = 1
    # df.loc[(df['score'] >= 0.50) & (df['score'] <= 0.75), 'score_class'] = 2
    # df.loc[(df['score'] >= 0.75), 'score_class'] = 3

    df.loc[(df['score'] >= 0) & (df['score'] < 0.7), 'score_class'] = 0
    df.loc[(df['score'] >= 0.7), 'score_class'] = 1

    df['new_score'] = 0
    df.loc[(df['stock_days_class'] == 1) & (df['margin_class'] == 1), 'new_score'] = 1
    df.drop(['stock_days_class', 'margin_class'], axis=1, inplace=True)

    new_targets_created = ['stock_class1', 'stock_class2', 'margem_class1', 'score_class', 'new_score']
    return new_targets_created


def dataset_split(df, target, oversample=0):
    print('Splitting dataset...')

    df_train, df_test = train_test_split(df, stratify=df[target], random_state=2)  # This ensures that the classes are evenly distributed by train/test datasets; Default split is 0.75/0.25 train/test

    df_train_y = df_train[target]
    df_train_x = df_train.drop(target, axis=1)

    df_test_y = df_test[target]
    df_test_x = df_test.drop(target, axis=1)

    # print('train_x', df_train_x.shape, 'test_x', df_test_x.shape)

    if oversample:
        print('Oversampling small classes...')
        df_train_x, df_train_y = oversample_data(df_train_x, df_train_y)

    return df_train_x, df_train_y, df_test_x, df_test_y


def tag(target, oversample, approach, score, feature_selection_check, feature_selection_criteria, model=None, sales_place=0):

    file_name = str(approach) + '_' + str(model) + '_target_' + str(target[0]) + '_scoring_' + str(score)
    if sales_place:
        file_name += '_' + str(sales_place)
    if oversample:
        file_name += '_oversample'
    if type(feature_selection_check) == int:
        file_name += '_' + str(feature_selection_check) + '_features_' + str(feature_selection_criteria.__name__)

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

    file_name = 'feature_importance_graph_'
    file_name += name

    if os.path.isfile('output/' + file_name + 'png'):
        os.remove('output/' + file_name + '.png')

    d = {'feature': features, 'importance': feature_importance}
    feat_importance_df = pd.DataFrame(data=d)
    feat_importance_df.sort_values(ascending=False, by='importance', inplace=True)

    # top_features = feat_importance_df[feat_importance_df['importance'] > 0.01]
    top_features = feat_importance_df.head(10)
    # print(top_features)

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
    prediction_test = best_model.predict(test_x)

    model.grid_performance(prediction=prediction_trainer, y=train_y)
    print('Train:', '\n', 'Micro:', model.micro, '\n', 'Macro:', model.macro, '\n', 'Accuracy:', model.accuracy, '\n', 'Class Report', '\n', model.class_report)
    model.grid_performance(prediction=prediction_test, y=test_y)
    print('Test:', '\n', 'Micro:', model.micro, '\n', 'Macro:', model.macro, '\n', 'Accuracy:', model.accuracy, '\n', 'Class Report', '\n', model.class_report)

    plot_confusion_matrix(test_y, best_model, prediction_test, name)
    tn, fp, fn, tp = plot_confusion_matrix(test_y, best_model, prediction_test, name, normalization=1)
    # tn, fp, fn, tp = confusion_matrix(test_y, prediction_test).ravel()
    # print('Value Counts:', Counter(prediction_test))
    print('TN:', tn)
    print('FP:', fp)
    print('FN:', fn)
    print('TP:', tp)
    specificity = tn / (tn + fp)
    print('Specificity:', specificity)

    class_report_csv(model.class_report, name)
    performance_metrics_csv(model.micro, model.macro, model.accuracy, specificity, name, start)

    return prediction_trainer, prediction_test


def performance_metrics_csv(micro, macro, accuracy, specificity, name, start):

    file_name = 'performance_evaluation_'
    file_name += name

    if os.path.isfile('output/' + file_name + '.csv'):
        os.remove('output/' + file_name + '.csv')

    # header = 0
    with open('output/' + file_name + '.csv', 'a', newline='') as csvfile:
        fieldnames = ['micro_f1', 'macro_f1', 'accuracy', 'specificity', 'running_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'micro_f1': micro, 'macro_f1': macro, 'accuracy': accuracy, 'specificity': specificity, 'running_time': (time.time() - start)})


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
