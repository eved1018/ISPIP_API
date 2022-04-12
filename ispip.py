import pandas as pd
import os
from sklearn.metrics import auc, matthews_corrcoef, f1_score, precision_recall_curve, roc_curve
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import scipy.stats

def main(df,feature_cols,annotated_col,predicted_col, proteins, xg = True):
    df = predict(df, feature_cols, "input","model",xg)
    results_df, roc_curve_data, pr_curve_data, bin_frame, fscore_mcc_by_protein, stats_df = postprocess(df, predicted_col, annotated_col)
    df_saver(results_df, "results", "output/")
    df_saver(bin_frame, "bin_frame", "output/")
    df_saver(fscore_mcc_by_protein, "fscore_mcc_by_protein","output/")
    # visualization(roc_curve_data, pr_curve_data, None, df, feature_cols,annotated_col, predicted_col, df, bin_frame, args_container)
    return

def df_saver(df, name, output_path_dir):
    out = os.path.join(output_path_dir, f'{name}.csv')
    df.to_csv(out)
    return


def predict(df, feature_cols, input_folder_path, model_name, xg, models=None) -> pd.DataFrame:
    if models is not None:
        rf_model, linreg_model, logreg_model, xgboost_model = models
    else:
        rf_model = joblib.load(f"models/RF_{model_name}.joblib")
        logreg_model = joblib.load(
            f"models/Logit_{model_name}.joblib")
        linreg_model = joblib.load(
            f"models/LinRegr_{model_name}.joblib")
        xgboost_model = joblib.load(
            f"models/XGB_{model_name}.joblib") if xg else None
    out_df: pd.DataFrame = randomforest_predict_from_trained_model(
        df, feature_cols, rf_model)
    out_df: pd.DataFrame = logreg_predict_from_trained_model(
        out_df, feature_cols, logreg_model)
    out_df: pd.DataFrame = linreg_predict_from_trained_model(
        out_df, feature_cols, linreg_model)
    out_df: pd.DataFrame = xgboost_predict_from_trained_model(
        out_df, feature_cols, xgboost_model) if xg else df
    return out_df


def randomforest_predict_from_trained_model(df: pd.DataFrame, feature_cols, rf_model) -> pd.DataFrame:
    y_prob = rf_model.predict_proba(df[feature_cols])
    y_prob_interface = [p[1] for p in y_prob]
    df['randomforest'] = y_prob_interface
    return df


def logreg_predict_from_trained_model(df, feature_cols, logreg_model) -> pd.DataFrame:
    prediction = logreg_model.predict_proba(df[feature_cols])
    df["logisticregresion"] = [p[1] for p in prediction]
    return df


def linreg_predict_from_trained_model(df, feature_cols, linreg_model) -> pd.DataFrame:
    df["linearregression"] = linreg_model.predict(df[feature_cols])
    return df

def xgboost_predict_from_trained_model(df, feature_cols, xgboost_model) -> pd.DataFrame:
    df["xgboost"] = xgboost_model.predict(df[feature_cols])
    return df


def postprocess(test_frame, predicted_col, annotated_col, autocutoff = 15) -> tuple:
    proteins = test_frame.protein.unique()
    results = []
    roc_curve_data: list = []
    pr_curve_data: list = []
    fscore_mcc_by_protein = pd.DataFrame(index=proteins)
    cutoff_dict = {protein: autocutoff for protein in proteins}
    # make testframe and cutoff dict this self in class
    params = [(pred, cutoff_dict, test_frame, annotated_col)
              for pred in predicted_col]

    with ProcessPoolExecutor(max_workers=4) as exe:
        return_vals = exe.map(analyses, params)
        for return_val in return_vals:
            test_frame[f'{return_val[0]}_bin'] = return_val[1]
            fscore_mcc_by_protein[[
                f'{return_val[0]}_fscore', f'{return_val[0]}_mcc']] = return_val[2].values.tolist()

            results.append(return_val[3])
            roc_curve_data.append(return_val[4])
            pr_curve_data.append(return_val[5])

    result_df = pd.DataFrame(
        results, columns=['predictor', 'f-score', 'mcc', 'roc_auc', 'pr_auc'])
    stats_df = pd.DataFrame(index=predicted_col, columns=predicted_col)
    test_frame = test_frame.sort_values(by=annotated_col, ascending=False)
    for index in predicted_col:
        for column in predicted_col:
            if index == column:
                stats_df.loc[index, column] = index
            else:
                pval, test, auc_diff = statistics(
                    test_frame, annotated_col, index, column)
                stats_df.loc[index, column] = pval  # below diagnol
                stats_df.loc[column, index] = auc_diff  # above diagnol
    
    return result_df, roc_curve_data, pr_curve_data, test_frame, fscore_mcc_by_protein, stats_df


def analyses(params) -> tuple:
    pred, cutoff_dict, test_frame, annotated_col = params
    top = test_frame.sort_values(by=[pred], ascending=False).groupby((["protein"])).apply(
        lambda x: x.head(cutoff_dict[x.name])).index.get_level_values(1).tolist()

    test_frame[f'{pred}_bin'] = [
        1 if i in top else 0 for i in test_frame.index.tolist()]
    fscore_mcc_per_protein = test_frame.groupby((["protein"])).apply(
        lambda x: fscore_mcc(x, annotated_col, pred))

    fscore, mcc = fscore_mcc(test_frame, annotated_col, pred)
    roc_and_pr_dic = roc_and_pr(test_frame, annotated_col, pred)

    results_list = [pred, fscore, mcc,
                    roc_and_pr_dic["roc_auc"], roc_and_pr_dic["pr_auc"]]
    roclist = [pred, roc_and_pr_dic["fpr"], roc_and_pr_dic["tpr"],
               roc_and_pr_dic["roc_auc"], roc_and_pr_dic["roc_thresholds"]]

    prlist = [pred, roc_and_pr_dic["recall"], roc_and_pr_dic["precision"],
              roc_and_pr_dic["pr_auc"], roc_and_pr_dic["pr_thresholds"]]

    return pred, test_frame[f'{pred}_bin'], fscore_mcc_per_protein, results_list, roclist, prlist


def fscore_mcc(x, annotated_col, pred) -> tuple:
    return f1_score(x[annotated_col], x[f'{pred}_bin']), matthews_corrcoef(x[annotated_col], x[f'{pred}_bin'])


def statistics(x, annotated_col, pred1, pred2) -> tuple:
    y_true = x[annotated_col]
    y1 = x[pred1]
    y2 = x[pred2]
    log10_pval, aucs = delong_roc_test(y_true, y1, y2)
    aucs = aucs.tolist()
    dauc = round(aucs[1] - aucs[0], 3)
    log10_pval = round(log10_pval.tolist()[0][0], 3)
    test = "signifigant" if log10_pval < -1.3 else "not significant"
    return log10_pval, test, dauc

def roc_and_pr(test_frame: pd.DataFrame, annotated_col, pred) -> dict:
    fpr, tpr, roc_thresholds = roc_curve(
        test_frame[annotated_col], test_frame[pred])
    roc_auc = round(auc(fpr, tpr), 3)
    precision, recall, pr_thresholds = precision_recall_curve(
        test_frame[annotated_col], test_frame[pred])
    pr_auc = round(auc(recall, precision), 3)
    roc_and_pr_dic: dict = {"fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds, "roc_auc": roc_auc,
                            "precision": precision, "recall": recall, "pr_thresholds": pr_thresholds, "pr_auc": pr_auc}
    return roc_and_pr_dic

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(
        aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack(
        (predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov), aucs

def data_preprocesss(df: pd.DataFrame) -> tuple:
    feature_cols: list = df.columns.tolist()[1:-1]
    annotated_col: str = df.columns.tolist()[-1]
    df["protein"] = [x.split('_')[1] for x in df['residue']]
    proteins: np.ndarray = df["protein"].unique()
    df.set_index('residue', inplace=True)
    df.isnull().any()  # double check use
    df = df.fillna(0)  # fill empty
    df = df[df['annotated'] != "ERROR"]
    df["annotated"] = pd.to_numeric(df["annotated"])
    return df, feature_cols, annotated_col, proteins

def cli():
    files = [i for i in os.listdir("input")]
    filename = input(f"please choose an input file {files}: ")
    df = pd.read_csv("input/" + filename)
    df, feature_cols, annotated_col, proteins = data_preprocesss(df)
    predicted_cols = feature_cols + ['logisticregresion', "linearregression",'randomforest', "xgboost"]
    return df, feature_cols, annotated_col,predicted_cols,  proteins 


if __name__ == '__main__':
    df, feature_cols, annotated_col, predicted_cols, proteins = cli()
    main(df, feature_cols, annotated_col, predicted_cols, proteins)
    