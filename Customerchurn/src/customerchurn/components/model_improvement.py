from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.utils import read_yaml, save_json, save_object, load_numpy
from src.customerchurn.mlflow_utils import setup_mlflow

import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


@dataclass
class ModelImprovementConfig:
    config_path: str = os.path.join("configs", "config.yaml")
    leaderboard_path: str = os.path.join("artifacts", "metrics", "leaderboard.csv")
    best_metrics_path: str = os.path.join("artifacts", "metrics", "best_model_metrics.json")
    best_model_path: str = os.path.join("artifacts", "models", "best_model.pkl")


class ModelImprovement:
    def __init__(self, config: ModelImprovementConfig = ModelImprovementConfig()):
        self.config = config
        self.cfg = read_yaml(self.config.config_path)

        # MLflow setup should NEVER block training
        try:
            self.ml_enabled = setup_mlflow(self.cfg)
        except Exception as e:
            logging.info(f"MLflow setup failed, continuing without MLflow. Reason: {e}")
            self.ml_enabled = False

        exp_cfg = self.cfg.get("experiments", {})
        self.cv_fold = int(exp_cfg.get("cv_fold", 5))
        self.random_state = int(exp_cfg.get("random_state", 42))

    def _is_sparse(self, X):
        return (sp is not None) and sp.issparse(X)

    def _best_threshold_by_f1(self, y_true, y_prob):
        best = {"threshold": 0.5, "f1": -1, "precision": 0, "recall": 0}
        for t in np.linspace(0.05, 0.95, 19):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best["f1"]:
                best = {
                    "threshold": float(t),
                    "f1": float(f1),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                }
        return best

    def _evaluate_probs(self, y_true, y_prob, threshold):
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "threshold": float(threshold),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            }
        }

    def _get_candidates(self):
        candidates = []

        # 1) Logistic Regression 
        for C in [0.1, 1.0, 5.0, 10.0]:
            candidates.append((
                f"logreg_C{C}",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
            ))

        # 2) SGD Logistic
        for alpha in [1e-4, 1e-3, 1e-2]:
            candidates.append((
                f"sgd_log_alpha{alpha}",
                SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    random_state=self.random_state
                )
            ))

        # 3) RandomForest
        candidates.append((
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                random_state=self.random_state,
                class_weight="balanced_subsample",
                n_jobs=1
            )
        ))

        # 4) GradientBoosting
        candidates.append((
            "grad_boost",
            GradientBoostingClassifier(random_state=self.random_state)
        ))

        return candidates

    def initiate_model_improvement(self, X_train_path, X_test_path, y_train_path, y_test_path):
        try:
            logging.info("======= MODEL IMPROVEMENT STARTED =======")

            X_train = load_numpy(X_train_path)
            X_test = load_numpy(X_test_path)
            y_train = load_numpy(y_train_path)
            y_test = load_numpy(y_test_path)

            # Ensure y is 1D
            y_train = np.asarray(y_train).astype(int).ravel()
            y_test = np.asarray(y_test).astype(int).ravel()


            skf = StratifiedKFold(n_splits=self.cv_fold, shuffle=True, random_state=self.random_state)

            rows = []
            best_name = None
            best_score = -1
            best_threshold = 0.5
            best_estimator = None

            for name, est in self._get_candidates():

                if self._is_sparse(X_train) and name in {"grad_boost"}:
                    logging.info(f"Skipping {name} (does not support sparse input well).")
                    continue

                logging.info(f"CV evaluating: {name}")

                if self.ml_enabled:
                    with mlflow.start_run(run_name=name):
                        mlflow.log_param("model_name", name)
                        mlflow.log_param("cv_folds", self.cv_fold)
                        try:
                            mlflow.log_params(est.get_params())
                        except Exception:
                            pass

                        oof_prob = cross_val_predict(
                            est, X_train, y_train,
                            cv=skf, method="predict_proba", n_jobs=None
                        )[:, 1]

                        thr_info = self._best_threshold_by_f1(y_train, oof_prob)
                        metrics = self._evaluate_probs(y_train, oof_prob, thr_info["threshold"])

                        mlflow.log_metric("cv_roc_auc", metrics["roc_auc"])
                        mlflow.log_metric("cv_pr_auc", metrics["pr_auc"])
                        mlflow.log_metric("cv_f1", metrics["f1"])
                        mlflow.log_metric("cv_precision", metrics["precision"])
                        mlflow.log_metric("cv_recall", metrics["recall"])
                        mlflow.log_metric("cv_best_threshold", metrics["threshold"])
                else:
                    oof_prob = cross_val_predict(
                        est, X_train, y_train,
                        cv=skf, method="predict_proba", n_jobs=None
                    )[:, 1]

                    thr_info = self._best_threshold_by_f1(y_train, oof_prob)
                    metrics = self._evaluate_probs(y_train, oof_prob, thr_info["threshold"])

                row = {
                    "model": name,
                    "cv_roc_auc": metrics["roc_auc"],
                    "cv_pr_auc": metrics["pr_auc"],
                    "cv_best_thr": metrics["threshold"],
                    "cv_f1": metrics["f1"],
                    "cv_precision": metrics["precision"],
                    "cv_recall": metrics["recall"],
                }
                rows.append(row)

                score = metrics["pr_auc"]
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_threshold = thr_info["threshold"]
                    best_estimator = est

            leaderboard = pd.DataFrame(rows).sort_values(["cv_pr_auc", "cv_roc_auc"], ascending=False)
            os.makedirs(os.path.dirname(self.config.leaderboard_path), exist_ok=True)
            leaderboard.to_csv(self.config.leaderboard_path, index=False)

            if best_estimator is None:
                raise ValueError("No models evaluated successfully. Check sparse/dense compatibility.")

            logging.info(f"Best model from CV: {best_name} | CV PR-AUC={best_score:.4f} | thr={best_threshold:.2f}")

            # Fit best on full train, evaluate on test once
            best_estimator.fit(X_train, y_train)
            test_prob = best_estimator.predict_proba(X_test)[:, 1]
            test_metrics = self._evaluate_probs(y_test, test_prob, best_threshold)

            payload = {
                "best_model": best_name,
                "chosen_by": "cv_pr_auc",
                "best_threshold_from_cv": float(best_threshold),
                "test_metrics": test_metrics,
                "leaderboard_path": self.config.leaderboard_path,
            }

            save_json(self.config.best_metrics_path, payload)
            save_object(self.config.best_model_path, best_estimator)

            # Log final best run + artifacts
            if self.ml_enabled:
                with mlflow.start_run(run_name=f"BEST_{best_name}"):
                    mlflow.log_param("best_model", best_name)
                    mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
                    mlflow.log_metric("test_pr_auc", test_metrics["pr_auc"])
                    mlflow.log_metric("test_f1", test_metrics["f1"])
                    mlflow.log_metric("test_precision", test_metrics["precision"])
                    mlflow.log_metric("test_recall", test_metrics["recall"])
                    mlflow.log_metric("threshold_used", best_threshold)

                    mlflow.log_artifact(self.config.leaderboard_path)
                    mlflow.log_artifact(self.config.best_metrics_path)
                    mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            logging.info(f"Leaderboard saved: {self.config.leaderboard_path}")
            logging.info(f"Best metrics saved: {self.config.best_metrics_path}")
            logging.info(f"Best model saved: {self.config.best_model_path}")
            logging.info("======= MODEL IMPROVEMENT COMPLETED =======")

            return self.config.best_model_path, self.config.best_metrics_path, self.config.leaderboard_path

        except Exception as e:
            raise CustomException(e, sys)
