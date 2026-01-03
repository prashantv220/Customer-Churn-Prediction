# -*- coding: utf-8 -*-
"""
Customer Churn Prediction Orchestrator
Refactored from monolithic notebook into modular, orchestrator-based pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

np.random.seed(42)
sns.set_style("whitegrid")

CONFIG = {
    "test_size": 0.15,
    "val_size": 0.1765,
    "random_state": 42,
    "cost_fn": 100,
    "cost_fp": 10,
    "budget": 50000,
    "retention_lift": 0.35,
    "data_path": "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"
}

def summarize_split(y, name):
    """Print dataset split statistics."""
    y = np.asarray(y)
    churn_rate = y.mean()
    print(f"{name}: n={len(y)} | churn_rate={churn_rate:.3f} | pos={y.sum()} neg={(1-y).sum()}")

def expected_cost(y_true, y_pred, cost_fn=100, cost_fp=10):
    """Calculate expected cost from confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * cost_fn + fp * cost_fp

def find_best_threshold(y_true, y_prob, cost_fn=100, cost_fp=10, grid=None):
    """Find threshold that minimizes expected cost."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    
    best = {"threshold": None, "cost": np.inf, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    
    for t in grid:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = fn * cost_fn + fp * cost_fp
        
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    
    return best

def plot_pr_roc(y_true, y_prob, title_prefix="Model"):
    """Plot ROC and Precision-Recall curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax[0].set_title(f"{title_prefix} ROC")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend()
    
    ax[1].plot(recall, precision, label=f"AP={ap:.3f}")
    ax[1].set_title(f"{title_prefix} Precision-Recall")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

def report_at_threshold(name, y_true, y_prob, thr, cost_fn=100, cost_fp=10):
    """Generate classification report at specific threshold."""
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    cost = expected_cost(y_true, y_hat, cost_fn, cost_fp)
    
    print(f"\n{name} @ thr={thr:.2f}")
    print("Confusion:", {"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    print("Expected cost:", cost)
    print(classification_report(y_true, y_hat, digits=3))

# ============================================================================
# STAGE 1: DATA LOADING & EXPLORATION
# ============================================================================

class DataLoadingStage:
    """Load and perform initial EDA."""
    
    def execute(self, config):
        print("\n" + "="*70)
        print("STAGE 1: DATA LOADING & EXPLORATION")
        print("="*70)
        
        # Load dataset
        df = pd.read_csv(config["data_path"])
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nChurn distribution:")
        print(df["Churn"].value_counts(dropna=False))
        
        # Quick EDA visuals
        self._plot_eda(df)
        
        return df
    
    def _plot_eda(self, df):
        """Generate exploratory plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.histplot(data=df, x="tenure", hue="Churn", bins=30, element="step", 
                     stat="density", common_norm=False, ax=axes[0, 0])
        axes[0, 0].set_title("Tenure distribution by churn")
        
        sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 1])
        axes[0, 1].set_title("Contract type vs churn")
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0])
        axes[1, 0].set_title("MonthlyCharges vs churn")
        
        sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[1, 1])
        axes[1, 1].set_title("Payment method vs churn")
        axes[1, 1].tick_params(axis='x', rotation=20)
        
        plt.tight_layout()
        plt.show()
        
        print("\nKey hypotheses:")
        print("- Lower tenure => higher churn probability")
        print("- Month-to-month contracts => higher churn")
        print("- Higher MonthlyCharges => higher churn")
        print("- E-check payment method => higher churn")

# ============================================================================
# STAGE 2: DATA PREPROCESSING & SPLITTING
# ============================================================================

class PreprocessingStage:
    """Handle feature engineering and train/val/test split."""
    
    def execute(self, df, config):
        print("\n" + "="*70)
        print("STAGE 2: PREPROCESSING & SPLITTING")
        print("="*70)
        
        # Extract target
        y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
        
        # Drop ID columns
        X = df.drop(columns=["Churn"])
        for col in ["customerID", "customerId", "CustomerID"]:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # Coerce TotalCharges to numeric
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        
        # Identify feature types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        
        print(f"Numeric columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical columns ({len(cat_cols)}): {cat_cols[:5]}... (showing first 5)")
        
        # Build preprocessing pipeline
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols)
            ]
        )
        
        # Stratified train/val/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["random_state"], 
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config["val_size"], random_state=config["random_state"],
            stratify=y_temp
        )
        
        summarize_split(y_train, "Train")
        summarize_split(y_val, "Val")
        summarize_split(y_test, "Test")
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "preprocess": preprocess,
            "num_cols": num_cols,
            "cat_cols": cat_cols
        }

# ============================================================================
# STAGE 3: MODEL TRAINING
# ============================================================================

class ModelTrainingStage:
    """Train baseline and advanced models."""
    
    def execute(self, data, config):
        print("\n" + "="*70)
        print("STAGE 3: MODEL TRAINING")
        print("="*70)
        
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        preprocess = data["preprocess"]
        
        # Logistic Regression variants
        print("\n--- Logistic Regression (Baseline) ---")
        lr_plain = Pipeline(steps=[
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
        ])
        
        lr_bal = Pipeline(steps=[
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"))
        ])
        
        lr_plain.fit(X_train, y_train)
        lr_bal.fit(X_train, y_train)
        
        val_prob_plain = lr_plain.predict_proba(X_val)[:, 1]
        val_prob_bal = lr_bal.predict_proba(X_val)[:, 1]
        
        print(f"LR plain ROC-AUC: {roc_auc_score(y_val, val_prob_plain):.4f}")
        print(f"LR balanced ROC-AUC: {roc_auc_score(y_val, val_prob_bal):.4f}")
        
        plot_pr_roc(y_val, val_prob_plain, "LogReg plain")
        plot_pr_roc(y_val, val_prob_bal, "LogReg balanced")
        
        # XGBoost with cost-sensitive learning
        print("\n--- XGBoost (Advanced) ---")
        pos = y_train.sum()
        neg = (1 - y_train).sum()
        scale_pos_weight = neg / pos
        
        xgb = Pipeline(steps=[
            ("prep", preprocess),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                min_child_weight=1.0,
                random_state=config["random_state"],
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            ))
        ])
        
        xgb.fit(X_train, y_train)
        val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
        
        print(f"XGB ROC-AUC: {roc_auc_score(y_val, val_prob_xgb):.4f}")
        plot_pr_roc(y_val, val_prob_xgb, "XGBoost")
        
        return {
            "lr_plain": lr_plain,
            "lr_bal": lr_bal,
            "xgb": xgb,
            "val_prob_plain": val_prob_plain,
            "val_prob_bal": val_prob_bal,
            "val_prob_xgb": val_prob_xgb
        }

# ============================================================================
# STAGE 4: THRESHOLD OPTIMIZATION
# ============================================================================

class ThresholdOptimizationStage:
    """Find cost-optimal decision thresholds."""
    
    def execute(self, models, data, config):
        print("\n" + "="*70)
        print("STAGE 4: THRESHOLD OPTIMIZATION")
        print("="*70)
        
        y_val = data["y_val"]
        val_prob_bal = models["val_prob_bal"]
        val_prob_xgb = models["val_prob_xgb"]
        
        cost_fn = config["cost_fn"]
        cost_fp = config["cost_fp"]
        
        best_lr = find_best_threshold(y_val, val_prob_bal, cost_fn=cost_fn, cost_fp=cost_fp)
        best_xgb = find_best_threshold(y_val, val_prob_xgb, cost_fn=cost_fn, cost_fp=cost_fp)
        
        print(f"\nBest threshold (LR balanced): {best_lr['threshold']:.4f} | Cost: {best_lr['cost']:.0f}")
        print(f"Best threshold (XGB): {best_xgb['threshold']:.4f} | Cost: {best_xgb['cost']:.0f}")
        
        # Compare default vs optimized thresholds
        print("\n--- LR Balanced @ Default (0.50) ---")
        report_at_threshold("LR balanced", y_val, val_prob_bal, 0.50, cost_fn, cost_fp)
        
        print("\n--- LR Balanced @ Optimal ---")
        report_at_threshold("LR balanced", y_val, val_prob_bal, best_lr["threshold"], cost_fn, cost_fp)
        
        print("\n--- XGB @ Default (0.50) ---")
        report_at_threshold("XGB", y_val, val_prob_xgb, 0.50, cost_fn, cost_fp)
        
        print("\n--- XGB @ Optimal ---")
        report_at_threshold("XGB", y_val, val_prob_xgb, best_xgb["threshold"], cost_fn, cost_fp)
        
        # Sensitivity analysis
        print("\n--- Sensitivity Analysis (varying cost ratios) ---")
        ratios = [2, 5, 10, 20, 50]
        sens = []
        
        for r in ratios:
            b = find_best_threshold(y_val, val_prob_xgb, cost_fn=10*r, cost_fp=10)
            sens.append((r, b["threshold"], b["cost"]))
        
        sens_df = pd.DataFrame(sens, columns=["FN_to_FP_ratio", "best_threshold", "min_cost"])
        print(sens_df.to_string(index=False))
        
        return {
            "best_lr": best_lr,
            "best_xgb": best_xgb,
            "sensitivity": sens_df
        }

# ============================================================================
# STAGE 5: RETENTION STRATEGY SIMULATION
# ============================================================================

class RetentionStrategyStage:
    """Simulate retention offers and budget allocation."""
    
    def execute(self, models, data, config):
        print("\n" + "="*70)
        print("STAGE 5: RETENTION STRATEGY SIMULATION")
        print("="*70)
        
        X_val = data["X_val"]
        y_val = data["y_val"]
        val_prob_xgb = models["val_prob_xgb"]
        
        # Build simulation dataframe
        val_df = X_val.copy()
        val_df["y_true"] = y_val.values
        val_df["p_churn"] = val_prob_xgb
        
        # Customer value proxy
        if "TotalCharges" in val_df.columns:
            val_df["value"] = pd.to_numeric(val_df["TotalCharges"], errors="coerce").fillna(0)
        else:
            val_df["value"] = val_df["MonthlyCharges"] * val_df["tenure"]
        
        # Risk buckets
        val_df["risk_bucket"] = pd.cut(
            val_df["p_churn"],
            bins=[-np.inf, 0.40, 0.70, np.inf],
            labels=["low", "medium", "high"]
        )
        
        # Intervention policy
        budget = config["budget"]
        offers = {"high": 30, "medium": 15, "low": 0}
        
        val_df["score_for_targeting"] = val_df["p_churn"] * val_df["value"]
        
        # Sort and allocate budget
        val_sorted = val_df.sort_values("score_for_targeting", ascending=False).copy()
        val_sorted["offer_cost_unit"] = (
            val_sorted["risk_bucket"].astype(str).map(offers).fillna(0).astype(int)
        )
        
        val_sorted["targeted"] = 0
        spent = 0
        
        for idx, row in val_sorted.iterrows():
            c = int(row["offer_cost_unit"])
            if c <= 0:
                continue
            if spent + c <= budget:
                val_sorted.at[idx, "targeted"] = 1
                spent += c
        
        print(f"Budget: ${budget:,.0f} | Spent: ${spent:,.0f} | Targeted customers: {int(val_sorted['targeted'].sum())}")
        
        # Simulate retention effectiveness
        lift = config["retention_lift"]
        rng = np.random.default_rng(config["random_state"])
        
        val_sorted["saved_churn"] = (
            (val_sorted["targeted"] == 1) &
            (val_sorted["y_true"] == 1) &
            (rng.random(len(val_sorted)) < lift)
        ).astype(int)
        
        # Financial analysis
        val_sorted["offer_cost"] = val_sorted["offer_cost_unit"] * val_sorted["targeted"]
        val_sorted["gain"] = val_sorted["saved_churn"] * val_sorted["value"]
        
        total_gain = val_sorted["gain"].sum()
        total_cost = val_sorted["offer_cost"].sum()
        net_gain = total_gain - total_cost
        roi = total_gain / max(total_cost, 1)
        
        print(f"\nSimulation Results (Lift={lift:.0%}):")
        print(f"  Avoided churns: {int(val_sorted['saved_churn'].sum())}")
        print(f"  Total revenue gain: ${total_gain:,.2f}")
        print(f"  Total offer cost: ${total_cost:,.2f}")
        print(f"  Net gain: ${net_gain:,.2f}")
        print(f"  ROI: {roi:.2f}x")
        
        print("\nSample targeting decisions (top 10):")
        print(val_sorted[["p_churn", "value", "risk_bucket", "targeted", "saved_churn", "gain"]].head(10).to_string())
        
        return {
            "val_sorted": val_sorted,
            "metrics": {
                "avoided_churns": int(val_sorted["saved_churn"].sum()),
                "revenue_gain": float(total_gain),
                "offer_cost": float(total_cost),
                "net_gain": float(net_gain),
                "roi": float(roi)
            }
        }

# ============================================================================
# STAGE 6: MODEL INTERPRETABILITY
# ============================================================================

class InterpretabilityStage:
    """Generate feature importance and SHAP explanations."""
    
    def execute(self, models, data):
        print("\n" + "="*70)
        print("STAGE 6: MODEL INTERPRETABILITY")
        print("="*70)
        
        # Logistic Regression coefficients
        self._interpret_logistic_regression(models, data)
        
        # SHAP for XGBoost
        self._interpret_xgboost(models, data)
    
    def _interpret_logistic_regression(self, models, data):
        """Extract and visualize odds ratios."""
        print("\n--- Logistic Regression Odds Ratios ---")
        
        lr_bal = models["lr_bal"]
        num_cols = data["num_cols"]
        cat_cols = data["cat_cols"]
        
        # Extract feature names
        ohe = lr_bal.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([np.array(num_cols), cat_feature_names])
        
        coefs = lr_bal.named_steps["clf"].coef_.ravel()
        odds_ratio = np.exp(coefs)
        
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs,
            "odds_ratio": odds_ratio
        }).sort_values("odds_ratio", ascending=False)
        
        print("\nTop churn-increasing features (highest odds ratio):")
        print(coef_df.head(15).to_string(index=False))
        
        print("\nTop churn-decreasing features (lowest odds ratio):")
        print(coef_df.tail(15).sort_values("odds_ratio").to_string(index=False))
    
    def _interpret_xgboost(self, models, data):
        """Generate SHAP explanations."""
        print("\n--- SHAP Explainability (XGBoost) ---")
        
        xgb = models["xgb"]
        X_val = data["X_val"]
        X_train = data["X_train"]
        num_cols = data["num_cols"]
        cat_cols = data["cat_cols"]
        
        # Transform data
        X_train_enc = xgb.named_steps["prep"].fit_transform(X_train)
        X_val_enc = xgb.named_steps["prep"].transform(X_val)
        
        # Feature names
        ohe = xgb.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([np.array(num_cols), cat_feature_names])
        
        # SHAP analysis
        model = xgb.named_steps["clf"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_enc)
        
        # Global importance
        shap.summary_plot(shap_values, X_val_enc, feature_names=feature_names, show=False)
        plt.title("SHAP summary (global importance)")
        plt.tight_layout()
        plt.show()
        
        # Local explanation for sample customer
        idx = 0
        X_val_enc_dense = X_val_enc[idx].toarray().ravel() if hasattr(X_val_enc[idx], "toarray") else X_val_enc[idx]
        
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_val_enc_dense,
            feature_names=feature_names
        ), show=False)
        plt.title("SHAP waterfall (sample customer prediction)")
        plt.tight_layout()
        plt.show()

# ============================================================================
# STAGE 7: FINAL EVALUATION
# ============================================================================

class EvaluationStage:
    """Evaluate on held-out test set."""
    
    def execute(self, models, data, thresholds, config):
        print("\n" + "="*70)
        print("STAGE 7: FINAL EVALUATION (TEST SET)")
        print("="*70)
        
        X_test = data["X_test"]
        y_test = data["y_test"]
        xgb = models["xgb"]
        best_thr = thresholds["best_xgb"]["threshold"]
        
        test_prob = xgb.predict_proba(X_test)[:, 1]
        y_test_hat = (test_prob >= best_thr).astype(int)
        
        test_auc = roc_auc_score(y_test, test_prob)
        test_ap = average_precision_score(y_test, test_prob)
        test_cost = expected_cost(y_test, y_test_hat, config["cost_fn"], config["cost_fp"])
        
        print(f"\nTest ROC-AUC: {test_auc:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        print(f"Using threshold: {best_thr:.4f}")
        print(f"Expected cost: ${test_cost:,.0f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_test_hat))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_hat, digits=3))
        
        plot_pr_roc(y_test, test_prob, "XGBoost (Test Set)")
        
        return {
            "test_auc": test_auc,
            "test_ap": test_ap,
            "test_cost": test_cost
        }

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ChurnPredictionOrchestrator:
    """Main orchestrator that coordinates all pipeline stages."""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.artifacts = {}
    
    def run(self):
        """Execute full pipeline."""
        print("\n" + "="*80)
        print("CUSTOMER CHURN PREDICTION ORCHESTRATOR")
        print("="*80)
        
        # Stage 1: Load & Explore
        loader = DataLoadingStage()
        df = loader.execute(self.config)
        self.artifacts["df"] = df
        
        # Stage 2: Preprocess & Split
        preprocessor = PreprocessingStage()
        data = preprocessor.execute(df, self.config)
        self.artifacts["data"] = data
        
        # Stage 3: Train Models
        trainer = ModelTrainingStage()
        models = trainer.execute(data, self.config)
        self.artifacts["models"] = models
        
        # Stage 4: Optimize Thresholds
        threshold_opt = ThresholdOptimizationStage()
        thresholds = threshold_opt.execute(models, data, self.config)
        self.artifacts["thresholds"] = thresholds
        
        # Stage 5: Retention Strategy
        retention = RetentionStrategyStage()
        retention_results = retention.execute(models, data, self.config)
        self.artifacts["retention"] = retention_results
        
        # Stage 6: Interpretability
        interp = InterpretabilityStage()
        interp.execute(models, data)
        
        # Stage 7: Final Evaluation
        evaluator = EvaluationStage()
        test_results = evaluator.execute(models, data, thresholds, self.config)
        self.artifacts["test_results"] = test_results
        
        print("\n" + "="*80)
        print("ORCHESTRATION COMPLETE")
        print("="*80)
        
        return self.artifacts

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    orchestrator = ChurnPredictionOrchestrator(config=CONFIG)
    results = orchestrator.run()
    
    # Access results via results["data"], results["models"], etc.
    print("\n✓ Pipeline completed successfully!")
    print("Available artifacts:", list(results.keys()))
