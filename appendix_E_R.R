required_packages <- c(
  "readxl",
  "isotree",
  "car",
  "lmtest",
  "moments",
  "tseries",
  "ggplot2",
  "gridExtra"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages)) {
  install.packages(new_packages)
}

library(readxl)
library(isotree)
library(car)
library(lmtest)
library(moments)
library(tseries)
library(ggplot2)
library(gridExtra)


file_path <- "A:/پایان نامه/نگارش/دیتای پایان نامه/victory.xlsx"

df_victory <- read_excel(file_path)


target_col <- "۶. معدل ترم اول کارشناسی"

persian_features <- c(
  "۱. سن",
  "۲. جنسیت",
  "۳. وضعیت تاهل",
  "۴. وضعیت اقتصادی خانواده",
  "۵. معدل کل دبیرستان",
  "سبک زندگی",
  "سلامت جسمانی",
  "سلامت روانشناختی"
)


df_v <- df_victory[
  complete.cases(df_victory[, c(target_col, persian_features)]),
]


X_v <- as.matrix(df_v[, persian_features])

set.seed(42)

iso_model <- isolation.forest(
  X_v,
  ntrees = 200,
  nthreads = 1
)

scores <- predict(iso_model, X_v)

threshold <- quantile(scores, 1 - 0.051)

clean_idx <- scores <= threshold

df_clean <- df_v[clean_idx, ]

cat(
  "Number of observations after Isolation Forest cleaning (N =",
  nrow(df_clean),
  ")\n"
)


df_model <- df_clean[, c(persian_features, target_col)]

colnames(df_model) <- c(
  "age",
  "gender",
  "marital",
  "econ",
  "hs_gpa",
  "lifestyle",
  "phys_health",
  "psych_health",
  "gpa_sem1"
)


model <- lm(
  gpa_sem1 ~ age + gender + marital + econ + hs_gpa +
    lifestyle + phys_health + psych_health,
  data = df_model
)

model_summary <- summary(model)

std_resid <- rstandard(model)

std_fitted <- as.numeric(scale(fitted(model)))


cat("\n--- 1. Durbin-Watson Test ---\n")

dw_res <- durbinWatsonTest(model)

print(dw_res)


cat("\n--- 2. Multicollinearity Analysis ---\n")

vif_vals <- vif(model)

tol_vals <- 1 / vif_vals

collinearity_df <- data.frame(
  Variable = names(vif_vals),
  Tolerance = round(tol_vals, 4),
  VIF = round(vif_vals, 4)
)

print(collinearity_df)


cat("\n--- 3. Normality of Residuals ---\n")

skew_val <- skewness(std_resid)

kurt_val <- kurtosis(std_resid) - 3

jb_res <- jarque.bera.test(std_resid)

cat(
  sprintf(
    "Skewness: %.4f | Excess Kurtosis: %.4f | Jarque-Bera p-value: %.4e\n",
    skew_val,
    kurt_val,
    jb_res$p.value
  )
)


df_diag <- data.frame(
  std_resid = std_resid,
  std_fitted = std_fitted
)


p_hist <- ggplot(df_diag, aes(x = std_resid)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 20,
    fill = "#5D99C6",
    color = "black",
    alpha = 0.85
  ) +
  stat_function(
    fun = dnorm,
    args = list(mean = 0, sd = 1),
    color = "red",
    linewidth = 1.1
  ) +
  labs(
    title = "Histogram of Standardized Residuals",
    x = "Standardized Residual",
    y = "Density"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 11
    ),
    panel.grid.major = element_line(
      linetype = "dashed",
      color = "grey80"
    )
  )


p_qq <- ggplot(df_diag, aes(sample = std_resid)) +
  stat_qq(
    shape = 21,
    fill = "#5D99C6",
    color = "black",
    size = 2.2,
    alpha = 0.8
  ) +
  stat_qq_line(
    color = "red",
    linewidth = 1.1
  ) +
  labs(
    title = "Normal Q-Q Plot of Standardized Residuals",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 11
    ),
    panel.grid.major = element_line(
      linetype = "dashed",
      color = "grey80"
    )
  )


fig_4_5 <- grid.arrange(
  p_hist,
  p_qq,
  ncol = 2
)

ggsave(
  "Figure_4_5_Normality.png",
  fig_4_5,
  width = 11,
  height = 5,
  dpi = 300
)


fig_4_6 <- ggplot(
  df_diag,
  aes(x = std_fitted, y = std_resid)
) +
  geom_point(
    shape = 21,
    fill = "#2E7D32",
    color = "#1B5E20",
    size = 2.8,
    alpha = 0.65
  ) +
  geom_hline(
    yintercept = 0,
    color = "#C62828",
    linetype = "dashed",
    linewidth = 1.1
  ) +
  labs(
    x = "",
    y = ""
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_line(
      linetype = "dashed",
      color = "grey85"
    ),
    panel.grid.minor = element_blank(),
    axis.text = element_text(
      size = 12,
      face = "bold"
    )
  )


ggsave(
  "Figure_4_6_Homoscedasticity.png",
  fig_4_6,
  width = 9,
  height = 5.5,
  dpi = 300
)


cat("\n--- 4. Outlier and Influential Observation Analysis ---\n")

predictors_mat <- as.matrix(df_model[, 1:8])

mahal_dist <- mahalanobis(
  predictors_mat,
  colMeans(predictors_mat),
  cov(predictors_mat)
)

crit_mahal <- qchisq(0.999, df = 8)

cat(
  "Critical Mahalanobis Distance (p=0.001, df=8):",
  round(crit_mahal, 3),
  "\n"
)

cat(
  "Maximum Observed Mahalanobis Distance:",
  round(max(mahal_dist), 3),
  "\n"
)

cooks_d <- cooks.distance(model)

cat(
  "Maximum Cook's Distance:",
  round(max(cooks_d), 4),
  "\n"
)


r_sq <- model_summary$r.squared

adj_r_sq <- model_summary$adj.r.squared

mult_r <- sqrt(r_sq)

rse <- model_summary$sigma


table_4_12 <- data.frame(
  Indicator = c(
    "Multiple Correlation Coefficient (R)",
    "Coefficient of Determination (R2)",
    "Adjusted R-squared",
    "Standard Error of Estimate",
    "Durbin-Watson Statistic",
    "Final Sample Size"
  ),
  Value = c(
    round(mult_r, 3),
    round(r_sq, 3),
    round(adj_r_sq, 3),
    round(rse, 3),
    round(dw_res$dw, 3),
    nrow(df_model)
  )
)

cat("\n--- Table 4-12: Model Fit Statistics ---\n")

print(table_4_12)


anova_res <- anova(model)

ss_reg <- sum(anova_res$`Sum Sq`[1:8])

df_reg <- 8

ms_reg <- ss_reg / df_reg

ss_res <- anova_res$`Sum Sq`[9]

df_res <- anova_res$Df[9]

ms_res <- anova_res$`Mean Sq`[9]

f_stat <- model_summary$fstatistic[1]

p_val_f <- pf(
  f_stat,
  df_reg,
  df_res,
  lower.tail = FALSE
)


table_4_13 <- data.frame(
  Source = c("Regression", "Residual", "Total"),
  Sum_of_Squares = round(
    c(ss_reg, ss_res, ss_reg + ss_res),
    3
  ),
  df = c(
    df_reg,
    df_res,
    df_reg + df_res
  ),
  Mean_Square = c(
    round(ms_reg, 3),
    round(ms_res, 3),
    NA
  ),
  F_Statistic = c(
    round(f_stat, 2),
    NA,
    NA
  ),
  P_Value = c(
    format.pval(p_val_f, digits = 3),
    NA,
    NA
  )
)

cat("\n--- Table 4-13: Analysis of Variance (ANOVA) ---\n")

print(table_4_13)


coef_mat <- model_summary$coefficients

df_scaled <- as.data.frame(scale(df_model))

model_std <- lm(
  gpa_sem1 ~ age + gender + marital + econ + hs_gpa +
    lifestyle + phys_health + psych_health,
  data = df_scaled
)

beta_std <- c(NA, coef(model_std)[-1])


table_4_14 <- data.frame(
  Variable = c(
    "Intercept",
    "Age",
    "Gender",
    "Marital Status",
    "Economic Status",
    "High School GPA",
    "Lifestyle",
    "Physical Health",
    "Mental Health"
  ),
  B = round(coef_mat[, 1], 3),
  Std_Error = round(coef_mat[, 2], 3),
  Beta = round(beta_std, 4),
  t_value = round(coef_mat[, 3], 3),
  Sig = round(coef_mat[, 4], 3),
  Tolerance = c(NA, round(tol_vals, 3)),
  VIF = c(NA, round(vif_vals, 3))
)

cat("\n--- Table 4-14: Multiple Regression Coefficients ---\n")

print(table_4_14)


b <- round(coef_mat[, 1], 3)

beta <- round(coef(model_std)[-1], 3)


cat("\n--- Unstandardized Regression Equation ---\n")

cat(
  sprintf(
    paste0(
      "GPA = %.3f + (%.3f)*Age + (%.3f)*Gender + ",
      "(%.3f)*Marital + (%.3f)*Econ + (%.3f)*HS_GPA + ",
      "(%.3f)*Lifestyle + (%.3f)*Phys + (%.3f)*Psych\n"
    ),
    b[1], b[2], b[3], b[4], b[5],
    b[6], b[7], b[8], b[9]
  )
)


cat("\n--- Standardized Regression Equation ---\n")

cat(
  sprintf(
    paste0(
      "Z_GPA = (%.3f)*Z_Age + (%.3f)*Z_Gender + ",
      "(%.3f)*Z_Marital + (%.3f)*Z_Econ + ",
      "(%.3f)*Z_HS_GPA + (%.3f)*Z_Lifestyle + ",
      "(%.3f)*Z_Phys + (%.3f)*Z_Psych\n"
    ),
    beta[1], beta[2], beta[3], beta[4],
    beta[5], beta[6], beta[7], beta[8]
  )
)


create_dist_plot <- function(
  data,
  var_name,
  plot_title,
  bar_color
) {

  x <- data[[var_name]]

  mu <- mean(x, na.rm = TRUE)

  sigma <- sd(x, na.rm = TRUE)

  sk <- skewness(x, na.rm = TRUE)

  kt <- kurtosis(x, na.rm = TRUE) - 3

  stats_text <- sprintf(
    "Mean: %.2f\nStd. Dev: %.2f\nSkewness: %.2f\nKurtosis: %.2f",
    mu,
    sigma,
    sk,
    kt
  )

  p <- ggplot(
    data,
    aes(x = .data[[var_name]])
  ) +

    geom_histogram(
      aes(y = after_stat(density)),
      bins = 15,
      fill = bar_color,
      color = "black",
      alpha = 0.75
    ) +

    stat_function(
      fun = dnorm,
      args = list(mean = mu, sd = sigma),
      color = "#000080",
      linewidth = 1.2
    ) +

    geom_vline(
      xintercept = mu,
      color = "red",
      linetype = "dashed",
      linewidth = 1.2
    ) +

    annotate(
      "label",
      x = -Inf,
      y = Inf,
      label = stats_text,
      hjust = -0.05,
      vjust = 1.05,
      size = 3.5,
      fontface = "bold",
      fill = "white",
      alpha = 0.85,
      color = "black"
    ) +

    labs(
      title = plot_title,
      x = "Score / Value",
      y = "Frequency"
    ) +

    theme_bw() +

    theme(
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 11
      ),
      panel.grid.major = element_line(
        linetype = "dashed",
        color = "grey80"
      ),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 10),
      axis.title = element_text(
        size = 10,
        face = "bold"
      )
    )

  return(p)
}


p1 <- create_dist_plot(
  df_model,
  "lifestyle",
  "Distribution: Lifestyle",
  "#5D99C6"
)

p2 <- create_dist_plot(
  df_model,
  "phys_health",
  "Distribution: Physical Health",
  "#58B16B"
)

p3 <- create_dist_plot(
  df_model,
  "psych_health",
  "Distribution: Mental Health",
  "#F39C12"
)

p4 <- create_dist_plot(
  df_model,
  "gpa_sem1",
  "Distribution: Academic Performance (GPA)",
  "#E74C3C"
)


fig_4panels <- grid.arrange(
  p1,
  p2,
  p3,
  p4,
  ncol = 2
)


ggsave(
  "Figure_Distributions_4Panels.png",
  fig_4panels,
  width = 12,
  height = 9,
  dpi = 300
)

cat("The four distribution histograms were successfully generated and saved.\n")