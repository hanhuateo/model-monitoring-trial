import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *


def performance_metrics(model, X, y, type):

    y_pred = model.predict(X)
    
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(data=cm, 
                        index=['True Negative: 0', 'True Positive: 1'], 
                        columns=['Pred Negative: 0', 'Pred Positive: 1'])
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')

    print("\n" "#=====#=====#===== Classification Report =====#=====#=====#")
    print(classification_report(y, y_pred))

    plt.savefig('./metrics_graphs/training_model_performance_metrics_{}.png'.format(type), format='png')

def plot_evaluation_curves(model, X_train, X_test, y_train, y_test):
    performance_metrics = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    for dataset, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        
        #=====#=====#=====#=====#=====
        # Compute metrics: Accuracy, F1, ROC AUC, Average Precision (AP)
        #=====#=====#=====#=====#=====
        performance_metrics[dataset] = {}
        
        y_pred = model.predict(X)
        y_pred_prob = model.predict_proba(X)[:, 1]
        
        # Accuracy
        accuracy = accuracy_score(y, y_pred)
        performance_metrics[dataset]['Accuracy'] = accuracy
        
        # F1
        f1 = f1_score(y, y_pred)
        performance_metrics[dataset]['F1'] = f1
        
        # ROC AUC
        roc_auc = roc_auc_score(y, y_pred_prob)
        performance_metrics[dataset]['ROC_AUC'] = roc_auc
        
        # Average Precision (AP)
        average_precision = average_precision_score(y, y_pred_prob)
        performance_metrics[dataset]['Average_Precision'] = average_precision
        
        #=====#=====#=====#=====#=====
        # Plot curves: F1 score, ROC, PRC
        #=====#=====#=====#=====#=====
        color = 'blue' if dataset == 'train' else 'orange'
        
        # F1 score
        thresholds = np.linspace(start=0, stop=1, num=100, endpoint=True)
        f1_scores = [f1_score(y, y_pred_prob >= th) for th in thresholds]
        max_f1_score_idx = np.argmax(f1_scores)
        
        ax = axs[0]
        ax.plot(thresholds, f1_scores, color=color, label=f"{dataset}, max={round(f1_scores[max_f1_score_idx], 2)} @ {round(thresholds[max_f1_score_idx], 2)}")
        # mark some thresholds
        for th in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            closest_threshold_idx = np.argmin(np.abs(thresholds-th))
            marker_color = 'red' 
            ax.plot(thresholds[closest_threshold_idx], f1_scores[closest_threshold_idx], color=marker_color, marker='X', markersize=7)        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('decision function threshold')
        ax.set_ylabel('F1')
        ax.set_title('F1 Score') 
        ax.legend(loc='lower center')
        

        # ROC
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f"{dataset}, ROC AUC={round(roc_auc, 2)}")
        # mark some thresholds
        for th in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            closest_threshold_idx = np.argmin(np.abs(thresholds-th))
            marker_color = 'red' 
            ax.plot(fpr[closest_threshold_idx], tpr[closest_threshold_idx], color=marker_color, marker='X', markersize=7)        
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve') 
        ax.legend(loc='lower center')
        
        # PRC
        precision, recall, thresholds = precision_recall_curve(y, y_pred_prob)
        
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f"{dataset}, Average Precision={round(average_precision, 2)}")
        # mark some thresholds
        for th in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            closest_threshold_idx = np.argmin(np.abs(thresholds-th))
            marker_color = 'red' 
            ax.plot(recall[closest_threshold_idx], precision[closest_threshold_idx], color=marker_color, marker='X', markersize=7)        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall Curve') 
        ax.legend(loc='lower center')
        
    
    df_performance_metrics = pd.DataFrame(performance_metrics).round(2)
    plt.savefig('./metrics_graphs/training_model_evaluation_curves.png', format='png')
    
def randomforestmodel(X_train_processed, y_train_processed):
    RF_clf = RandomForestClassifier(class_weight='balanced', random_state=42)

    param_grid = {'n_estimators': [100, 500, 900], 
                'max_features': ['auto', 'sqrt'],
                'max_depth': [2, 15, None], 
                'min_samples_split': [5, 10],
                'min_samples_leaf': [1, 4], 
                'bootstrap': [True, False]
                }
    
    RF_search = GridSearchCV(RF_clf, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    RF_search.fit(X_train_processed, y_train_processed)

    print(RF_search.best_params_)

    RF_clf = RandomForestClassifier(**RF_search.best_params_, class_weight='balanced', random_state=42)
    RF_clf.fit(X_train_processed, y_train_processed)
    dump(RF_clf, '../data/models/RF_clf.joblib')
    return RF_clf

def train_model(df):
    df = object_to_category(df)
    df = drop_columns(df)
    df = ordinal_encoding(df)
    X_train, X_test, y_train, y_test = split(df)
    X_train_processed,  X_test_processed, y_train_processed, y_test_processed = train_preprocessing(X_train, X_test, y_train, y_test)
    RF_clf = randomforestmodel(X_train_processed, y_train_processed)
    performance_metrics(RF_clf, X_train_processed, y_train_processed, "train")
    performance_metrics(RF_clf, X_test_processed, y_test_processed, "test")
    plot_evaluation_curves(RF_clf, X_train_processed, X_test_processed, y_train_processed, y_test_processed)