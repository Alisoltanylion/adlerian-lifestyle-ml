import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)


file_path = r"A:\پایان نامه\نگارش\دیتای پایان نامه\victory.xlsx"
df_victory = pd.read_excel(file_path)

target_col = "۶. معدل ترم اول کارشناسی"

persian_features = [
    "۱. سن",
    "۲. جنسیت",
    "۳. وضعیت تاهل",
    "۴. وضعیت اقتصادی خانواده",
    "۵. معدل کل دبیرستان",
    "سبک زندگی",
    "سلامت جسمانی",
    "سلامت روانشناختی",
]


df_v = df_victory.dropna(
    subset=[target_col] + persian_features
).copy()

X_v = df_v[persian_features]
y_v = df_v[target_col]

iso = IsolationForest(
    contamination=0.051,
    random_state=42
)

outliers = iso.fit_predict(X_v)

df_clean = df_v[outliers == 1].copy()
cols = persian_features + [target_col]


desc_stats = pd.DataFrame({
    "Mean": df_clean[cols].mean(),
    "Std": df_clean[cols].std(),
    "Skewness": df_clean[cols].skew(),
    "Kurtosis": df_clean[cols].kurtosis(),
})

print("=" * 70)
print(f"Descriptive Statistics (N={len(df_clean)})")
print("=" * 70)
print(desc_stats.round(4))


corr_matrix = df_clean[cols].corr(method="pearson")

print("\n" + "=" * 70)
print("Pearson Correlation Matrix")
print("=" * 70)
print(corr_matrix.round(4))


X_clean = df_clean[persian_features]
y_clean = df_clean[target_col]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean,
    y_clean,
    test_size=0.20,
    random_state=42
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


ols_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


svr_grid = GridSearchCV(
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ]),
    [
        {
            "model__kernel": ["linear"],
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "model__gamma": [
                "scale", "auto", 0.001, 0.01, 0.1
            ],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        },
    ],
    cv=kf,
    scoring="r2",
    n_jobs=-1,
).fit(X_train_c, y_train_c)


rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [2, 4, 6, 8, 10],
        "max_features": ["sqrt", "log2", 1.0],
    },
    cv=kf,
    scoring="r2",
    n_jobs=-1,
).fit(X_train_c, y_train_c)


xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42),
    {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    cv=kf,
    scoring="r2",
    n_jobs=-1,
).fit(X_train_c, y_train_c)


mlp_grid = GridSearchCV(
    Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            MLPRegressor(
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
            ),
        ),
    ]),
    {
        "model__activation": ["relu", "tanh", "identity"],
        "model__hidden_layer_sizes": [
            (4,), (8,), (16,), (8, 4), (16, 8)
        ],
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "model__learning_rate_init": [
            0.0005, 0.001, 0.005
        ],
    },
    cv=kf,
    scoring="r2",
    n_jobs=-1,
).fit(X_train_c, y_train_c)


print("=" * 70)
print("Optimal Hyperparameters")
print("=" * 70)

print(f"SVR:\n{svr_grid.best_params_}\n")
print(f"Random Forest:\n{rf_grid.best_params_}\n")
print(f"XGBoost:\n{xgb_grid.best_params_}\n")
print(f"MLP:\n{mlp_grid.best_params_}\n")
print("OLS:\nStandard Linear Regression")
print("=" * 70)


final_models = {
    "SVR": svr_grid.best_estimator_,
    "OLS": ols_pipe.fit(X_train_c, y_train_c),
    "MLP": mlp_grid.best_estimator_,
    "XGBoost": xgb_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
}


golden_results = []
predictions = {}

for name, model in final_models.items():

    model.fit(X_train_c, y_train_c)

    y_pred_tr = model.predict(X_train_c)
    y_pred_te = model.predict(X_test_c)

    predictions[name] = y_pred_te

    cv_scores = cross_validate(
        model,
        X_clean,
        y_clean,
        cv=kf,
        scoring="r2",
        n_jobs=-1
    )

    cv_r2_mean = cv_scores["test_score"].mean()

    r2_tr = r2_score(y_train_c, y_pred_tr)
    r2_te = r2_score(y_test_c, y_pred_te)

    delta = r2_tr - r2_te

    golden_results.append({
        "Model": name,
        "R2_Train": round(r2_tr, 4),
        "R2_Test": round(r2_te, 4),
        "R2_5Fold_CV": round(cv_r2_mean, 4),
        "Delta_R2": round(delta, 4),
        "RMSE_Test": round(
            np.sqrt(
                mean_squared_error(y_test_c, y_pred_te)
            ),
            4
        ),
        "MAE_Test": round(
            mean_absolute_error(y_test_c, y_pred_te),
            4
        ),
        "Overfitting_Status": (
            "Optimal (<=0.05)"
            if delta <= 0.05
            else "Overfitting (>0.05)"
        ),
    })


df_golden = pd.DataFrame(
    golden_results
).sort_values(
    by="R2_Test",
    ascending=False
)

print("=" * 70)
print(f"Machine Learning Model Comparison (N={len(df_clean)})")
print("=" * 70)
print(df_golden.to_string(index=False))


fig = plt.figure(figsize=(14, 19))

ax1 = fig.add_axes([0.08, 0.68, 0.38, 0.26])
ax2 = fig.add_axes([0.54, 0.68, 0.38, 0.26])
ax3 = fig.add_axes([0.08, 0.37, 0.38, 0.26])
ax4 = fig.add_axes([0.54, 0.37, 0.38, 0.26])
ax5 = fig.add_axes([0.31, 0.06, 0.38, 0.26])

axes_list = [
    (ax1, "OLS", predictions["OLS"]),
    (ax2, "Random Forest", predictions["Random Forest"]),
    (ax3, "XGBoost", predictions["XGBoost"]),
    (ax4, "MLP", predictions["MLP"]),
    (ax5, "SVR", predictions["SVR"]),
]


all_preds = np.concatenate(list(predictions.values()))

min_val = min(
    np.min(y_test_c),
    np.min(all_preds)
)

max_val = max(
    np.max(y_test_c),
    np.max(all_preds)
)


for ax, name, y_pred in axes_list:

    ax.scatter(
        y_test_c,
        y_pred,
        alpha=0.85,
        edgecolors="k",
        s=65,
        color="#0b5394",
        linewidth=0.8
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal Line ($y=x$)"
    )

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    ax.set_xlabel(
        "Actual GPA",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Predicted GPA",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_title(
        f"Actual vs Predicted: {name}",
        fontsize=11,
        fontweight="bold"
    )

    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)


plt.savefig(
    "actual_vs_predicted_all_models.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Actual vs Predicted plot saved.")


best_model_name = df_golden.iloc[0]["Model"]

print(f"Best Model: {best_model_name}")

best_estimator = final_models[best_model_name]


english_features = [
    "Age",
    "Gender",
    "Marital Status",
    "Economic Status",
    "High School GPA",
    "Lifestyle",
    "Physical Health",
    "Mental Health",
]


if (
    "SVR" in best_model_name
    or "MLP" in best_model_name
    or "OLS" in best_model_name
):

    scaler = best_estimator.named_steps["scaler"]
    model_obj = best_estimator.named_steps["model"]

    X_tr_sc = scaler.transform(X_train_c)
    X_te_sc = scaler.transform(X_test_c)

    def predict_wrapper(x):
        return model_obj.predict(x)

    bg = X_tr_sc[
        np.random.choice(
            X_tr_sc.shape[0],
            min(50, len(X_tr_sc)),
            replace=False
        )
    ]

    explainer = shap.KernelExplainer(
        predict_wrapper,
        bg
    )

    shap_vals = explainer.shap_values(X_te_sc)
    eval_features = X_te_sc

else:

    explainer = shap.TreeExplainer(best_estimator)

    shap_vals = explainer.shap_values(X_test_c)
    eval_features = X_test_c


plt.figure(figsize=(9, 5.2))

shap.summary_plot(
    shap_vals,
    eval_features,
    feature_names=english_features,
    plot_type="bar",
    show=False
)

ax = plt.gca()

bars = [
    p for p in ax.patches
    if p.get_width() > 0
]

total_width = sum(
    bar.get_width()
    for bar in bars
)

for bar in bars:

    w = bar.get_width()
    pct = (w / total_width) * 100

    ax.text(
        w + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="#1565C0"
    )


max_w = max(
    b.get_width()
    for b in bars
)

ax.set_xlim(0, max_w * 1.14)

ax.set_xlabel(
    "mean(|SHAP value|) (average impact on model output magnitude)",
    fontsize=11,
    fontweight="bold"
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=11
)

plt.tight_layout()

plt.savefig(
    "shap_bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(9, 5))

shap.summary_plot(
    shap_vals,
    eval_features,
    feature_names=english_features,
    plot_type="dot",
    show=False
)

plt.title(
    f"SHAP Beeswarm Plot ({best_model_name})",
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "shap_beeswarm_plot.png",
    dpi=300
)

plt.close()


print(
    f"SHAP plots for the best model ({best_model_name}) "
    "were successfully generated and saved."
)