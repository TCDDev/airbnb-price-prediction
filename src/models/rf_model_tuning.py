from sklearn.model_selection import GridSearchCV, train_test_split

from src.data.load import load_features_and_target
from src.models.pipelines import build_random_forest_pipeline
from src.utils.paths import METRICS_DIR

def tune():
    X, y = load_features_and_target

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

    # TODO: evaluate search.best_estimator_ on X_test once, then save metrics.


if __name__ == "__main__":
    tune()
