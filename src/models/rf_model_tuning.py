from sklearn.model_selection import GridSearchCV, train_test_split

from src.data.load import load_features_and_target
from src.models.pipelines import build_random_forest_pipeline
from src.utils.paths import FIGURES_DIR, METRICS_DIR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt


def tune():
    X, y = load_features_and_target()

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123,
    )

    pipeline = build_random_forest_pipeline(
        numerical_cols,
        categorical_cols,
        boolean_cols,
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(search.best_params_)
    print(-search.best_score_)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print("\n Best Model Test Set Results:")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"MAE:  {test_mae:.2f}")
    print(f"R²:   {test_r2:.2f}")
    
   # Feature Importance
    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = []
    feature_names.extend(numerical_cols)
    
    cat_transformer = preprocessor.named_transformers_["cat"]
    if "onehot" in cat_transformer.named_steps:
        onehot_cols = cat_transformer.named_steps["onehot"].get_feature_names_out()
        feature_names.extend(onehot_cols)
    else:
        feature_names.extend(categorical_cols)
        
    feature_names.extend(boolean_cols)

    importances = best_model.named_steps["model"].feature_importances_
    named_importances = pd.Series(importances, index=feature_names)
    top_5 = named_importances.sort_values(ascending=False).head(5)


    metrics_path = METRICS_DIR / "random_forest_tuned_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Tuned Random Forest Evaluation (Test Set)\n")
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"Best CV RMSE: {-search.best_score_:.4f}\n\n")
        f.write(f"Test RMSE: {test_rmse:.4f}\n")
        f.write(f"Test MAE:  {test_mae:.4f}\n")
        f.write(f"Test R²:   {test_r2:.4f}\n\n")
        
        f.write("Top 5 Feature Importances (Tuned Model)\n")
        for col, val in top_5.items():
            f.write(f"{col}: {val:.6f}\n")

    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal', label="Tuned RF Predictions")
    ideal_line = [y_test.min(), y_test.max()]
    plt.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label="Ideal Prediction")
    plt.title("Tuned Random Forest Regression: Actual vs. Predicted")
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    figure_path = FIGURES_DIR / "rf_tuned_actual_vs_predicted.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Importance plot
    plt.figure(figsize=(10, 6))
    top_10_features = named_importances.sort_values(ascending=False).head(10)
    top_10_features.sort_values(ascending=True).plot(kind='barh', color='teal')
    plt.title("Top 10 Feature Importances (Tuned Random Forest)")
    plt.xlabel("Relative Importance")
    plt.ylabel("Features")
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    feat_img_path = FIGURES_DIR / "rf_tuned_feature_importances.png"
    plt.savefig(feat_img_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    tune()