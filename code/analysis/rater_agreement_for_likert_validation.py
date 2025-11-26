import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, cohen_kappa_score
import pingouin as pg
import krippendorff
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json

ratings_file = "/localdisk1/PARK/park_vlm_finetuning/data/Likert_Human_Validation/validate_likert_score_combined.csv"

def pre_process_annotations():

    def expert_agreement(row):
        r1 = row["A1_likert_score"]
        r2 = row["A2_likert_score"]
        r3 = row["A3_likert_score"]

        if (r1==r2) or (r1==r3):
            return r1
        
        if r2==r3:
            return r2
        
        return int((r1+r2+r3)/3)

    df = pd.read_csv(ratings_file)
    
    annotator_names = ["A1", "A2", "A3"]
    df_annotators = {}
    for a_name in annotator_names:
        df = df.drop(columns=[f"{a_name}_likert_score"])
        a_file = f"/localdisk1/PARK/park_vlm_finetuning/data/Likert_Human_Validation/validate_likert_score_combined_{a_name}.csv"
        df_annotators[a_name] = pd.read_csv(a_file)
        df_annotators[a_name] = df_annotators[a_name][["row_id", f"{a_name}_likert_score"]]
        df = pd.merge(df, df_annotators[a_name], how="inner", on="row_id")

    df["gt_likert_score"] = df.apply(expert_agreement, axis=1)
    df.to_csv(ratings_file, index=False)
    return


def complete_agreement_rate(R1, R2):
    R1 = np.array(R1).astype(int)
    R2 = np.array(R2).astype(int)
    return np.mean(R1==R2)

def ordinal_classification_accuracy(y_true, y_pred, num_classes=5):
    """
    Computes the Ordinal Classification Accuracy (OCA).
    
    Parameters:
    - y_true: List or np.array of true labels
    - y_pred: List or np.array of predicted labels
    - num_classes: Total number of ordinal classes
    
    Returns:
    - OCA score (0 to 1, where 1 is perfect accuracy)
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    absolute_errors = np.abs(y_true - y_pred)  # Compute misclassification distances
    max_possible_error = (num_classes - 1) * len(y_true)  # Maximum possible error

    oca_score = 1 - (np.sum(absolute_errors) / max_possible_error)
    return oca_score

def RMSE(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

def QWK(R1, R2):
    R1 = np.array(R1).astype(int)
    R2 = np.array(R2).astype(int)
    return cohen_kappa_score(y1=R1, y2=R2, weights="quadratic")

class AgreementMetrics:
    def __init__(self, ratings, mode):
        self.mode = mode
        self.ratings = ratings

        if self.mode == "human-human":
            self.num_annotators = 3
            self.rater_names = ["A1", "A2", "A3"]
            self.rating_columns = [f"{a_name}_likert_score" for a_name in self.rater_names]

        elif self.mode == "human-AI":
            self.num_annotators = 3
            self.rater_names = ["A1", "A2", "A3"]
            self.rating_columns = [f"{a_name}_likert_score" for a_name in self.rater_names]
            self.gpt_rating_column = "gpt_likert_score"

        elif self.mode == "GT-AI":
            self.num_annotators = 1
            self.rater_names = ["gt"]
            self.rating_columns = [f"{a_name}_likert_score" for a_name in self.rater_names]
            self.gpt_rating_column = "gpt_likert_score"

        self.metrics = {
            "agreement_rate": [],
            "ordinal_accuracy": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "QWK": [],
            "alpha": None,
            "icc_single": None,
            "icc_single_ci": None,
            "icc_avg": None,
            "icc_avg_ci": None
        }

        self.metric_functions = {
            "agreement_rate": complete_agreement_rate,
            "ordinal_accuracy": ordinal_classification_accuracy,
            "MAE": mean_absolute_error,
            "MAPE": mean_absolute_percentage_error,
            "RMSE": RMSE,
            "QWK": QWK,
            "alpha": None,
            "icc_single": None,
            "icc_single_ci": None,
            "icc_avg": None,
            "icc_avg_ci": None
        }

    def compute_icc(self):
        if self.mode!="human-human":
            return None
        
        # Prepare lists for DataFrame creation
        exam = []
        judge = []
        rating = []

        # Iterate over each row and each rater column
        idx = 1  # Unique identifier for each row or "exam"
        for _, row in self.ratings.iterrows():
            for j, col in enumerate(self.rating_columns):
                exam.append(idx)
                judge.append(f'Rater_{j}')
                rating.append(row[col])
            idx += 1

        df_ratings = pd.DataFrame({'exam': exam, 'judge': judge, 'rating': rating})
        icc_results = pg.intraclass_corr(data=df_ratings, targets='exam', raters='judge', ratings='rating')

        return icc_results
    
    def compute_alpha(self):
        if self.mode!="human-human":
            return None
        
        # Prepare lists for DataFrame creation
        exam = []
        judge = []
        rating = []

        # Iterate over each row and each rater column
        idx = 1  # Unique identifier for each row or "exam"
        for _, row in self.ratings.iterrows():
            for j, col in enumerate(self.rating_columns):
                exam.append(idx)
                judge.append(f'Rater_{j}')
                rating.append(row[col])
            idx += 1

        df_ratings = pd.DataFrame({'exam': exam, 'judge': judge, 'rating': rating})

        # df has columns: 'subject', 'rater', 'rating'
        table = df_ratings.pivot(index='judge', columns='exam', values='rating')
        # Convert to numpy array (raters x items)
        data = table.to_numpy(dtype=float)
        alpha_interval = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement='interval'  # or 'ordinal'
        )
        return alpha_interval

    def compute_agreement_metrics(self):
        if self.mode == "human-human":
            first_rating_columns = self.rating_columns
        elif self.mode == "human-AI":
            first_rating_columns = [self.gpt_rating_column]
        elif self.mode == "GT-AI":
            first_rating_columns = [self.gpt_rating_column]

        for i, col1 in enumerate(first_rating_columns):
            R1 = self.ratings[col1]
            for j, col2 in enumerate(self.rating_columns):
                if self.mode=="human-human" and i==j:
                    continue
                R2 = self.ratings[col2]

                for metric in self.metrics:
                    if (self.metrics[metric] is not None) and (self.metric_functions[metric] is not None):
                        self.metrics[metric].append(self.metric_functions[metric](R1, R2))

                # R1 = np.array(R1).astype(int)
                # R2 = np.array(R2).astype(int)
                # cm = confusion_matrix(R1, R2)
                
                # # Convert to DataFrame for easy labels
                # labels = sorted(list(set(R1) | set(R2)))
                # cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                # fig, ax = plt.subplots(figsize=(6, 5))
                # im = ax.imshow(cm_df, cmap=plt.cm.Blues)

                # # Add labels
                # ax.set_xticks(np.arange(len(cm_df.columns)))
                # ax.set_yticks(np.arange(len(cm_df.index)))

                # ax.set_xticklabels(cm_df.columns)
                # ax.set_yticklabels(cm_df.index)

                # ax.set_xlabel(f"Rater {i}")
                # ax.set_ylabel(f"Rater {j}")

                # # Add values inside boxes
                # for ai in range(cm_df.shape[0]):
                #     for aj in range(cm_df.shape[1]):
                #         value = cm_df.iloc[ai, aj]
                #         ax.text(aj, ai, str(value),
                #                 ha='center', va='center',
                #                 color='red', fontsize=10)

                # plt.colorbar(im)
                # plt_dir = "/localdisk1/PARK/park_vlm_finetuning/plots/clinician_agreements"
                # os.makedirs(plt_dir, exist_ok=True)
                # plt.savefig(os.path.join(plt_dir, f"rater{i}_vs_rater{j}.png"), dpi=300, bbox_inches='tight')

        for metric in self.metrics:
            if (self.metrics[metric] is not None) and (len(self.metrics[metric])!=0):
                self.metrics[metric] = np.mean(self.metrics[metric]).item()

        icc_results = self.compute_icc()
        if icc_results is not None:
            self.metrics["icc_single"] = icc_results.loc[icc_results['Type'] == 'ICC3', 'ICC'].values[0].item()
            self.metrics["icc_single_ci"] = icc_results.loc[icc_results['Type'] == 'ICC3', 'CI95%'].values[0][0].item(), icc_results.loc[icc_results['Type'] == 'ICC3', 'CI95%'].values[0][1].item()
            self.metrics["icc_avg"] = icc_results.loc[icc_results['Type'] == 'ICC3k', 'ICC'].values[0].item()
            self.metrics["icc_avg_ci"] = icc_results.loc[icc_results['Type'] == 'ICC3k', 'CI95%'].values[0][0].item(), icc_results.loc[icc_results['Type'] == 'ICC3k', 'CI95%'].values[0][1].item()

        self.metrics["alpha"] = self.compute_alpha()

        return self.metrics
    
def main():
    ratings = pd.read_csv(ratings_file)
    likert_validation_metrics = {}

    agreement_metrics = AgreementMetrics(ratings, "human-human")
    metrics = agreement_metrics.compute_agreement_metrics()
    likert_validation_metrics["human-human"] = metrics
    print(likert_validation_metrics)

    agreement_metrics = AgreementMetrics(ratings, "human-AI")
    metrics = agreement_metrics.compute_agreement_metrics()
    likert_validation_metrics["human-AI"] = metrics
    print(likert_validation_metrics)

    agreement_metrics = AgreementMetrics(ratings, "GT-AI")
    metrics = agreement_metrics.compute_agreement_metrics()
    likert_validation_metrics["GT-AI"] = metrics
    print(likert_validation_metrics)

    results_base_path = "/localdisk1/PARK/park_vlm_finetuning/results"
    os.makedirs(results_base_path, exist_ok=True)
    results_path = os.path.join(results_base_path, "likert_conversion_validation.json")
    with open(results_path, "w") as f:
        json.dump(likert_validation_metrics, f, indent=4)

if __name__ == "__main__":
    pre_process_annotations()
    main()