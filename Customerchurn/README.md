## Customer churn Prediction ##

### Problem Statement  ###

Identifying the customer who will churn 


### Action  ###

What can be done to retained them?

### Model Card: Telco Customer Churn (Best Model) ###

#### Goal ####
Predict which customers are likely to churn so the business can target retention actions efficiently.

#### Data & Setup ####
Target: Churn (Yes = 1, No = 0)
Evaluation: Stratified K-Fold Cross-Validation
Selection metric: PR-AUC (better for imbalanced churn data than ROC-AUC)
Threshold tuning: chose probability threshold that maximized F1 on CV out-of-fold predictions

#### Model Selection Results (Cross-Validation) ####

Best models by CV PR-AUC:
GradientBoostingClassifier (Best by PR-AUC)
CV ROC-AUC: 0.8464
CV PR-AUC: 0.6592
Best threshold (by F1): 0.35
CV F1: 0.6333
Logistic Regression (Very close second)
CV ROC-AUC: ~0.845
CV PR-AUC: ~0.657
CV F1: ~0.637

#### ✅ Final chosen model: Gradient Boosting (chosen by CV PR-AUC) ####

### Final Test Performance (Hold-out Test Set) ###

Using threshold 0.35:
ROC-AUC: 0.8406
PR-AUC: 0.6563
F1: 0.6288
Precision: 0.5636
Recall: 0.7112
Confusion Matrix (Test)
TN: 829
FP: 206
FN: 108
TP: 266

### Interpretation ###

At threshold 0.35: The model catches 71% of churners (high recall = fewer missed churners).
About 56% of churn predictions are correct. There are 206 false alarms. 
This threshold is a good choice when missing churners is more expensive than contacting extra customers.

Recommendation: how to adjust threshold
Want fewer false positives (less outreach cost)? → increase threshold (e.g., 0.45–0.60)
Want catch more churners (maximize recall)? → decrease threshold (e.g., 0.25–0.35)

#### Artifacts Produced ####
artifacts/models/best_model.pkl -→ trained best model
artifacts/metrics/best_model_metrics.json -→ final test metrics + threshold
artifacts/metrics/leaderboard.csv -→ CV leaderboard for all candidate models