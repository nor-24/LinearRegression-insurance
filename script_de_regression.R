# ============================================================
# INSURANCE CHARGES ANALYSIS USING LINEAR REGRESSION
# ============================================================
# This script performs a complete statistical analysis:
# - Data loading
# - Data preprocessing
# - Linear regression modeling
# - Diagnostic tests
# - Outlier detection
# - Model comparison
# ============================================================

# ================================
# 1. DATA IMPORT
# ================================

# Load the dataset (make sure the file is in your working directory)
data <- read.csv("insurance.csv")

# Quick overview of the dataset
head(data)        # Display first rows
summary(data)     # Statistical summary

# Visualize the distribution of the target variable
hist(data$charges, main="Distribution of Charges", xlab="Charges")

# ================================
# 2. DATA PREPROCESSING
# ================================

# Convert categorical variables into factors
# This is required for linear regression in R
data$smoker <- as.factor(data$smoker)
data$sex <- as.factor(data$sex)
data$region <- as.factor(data$region)

# ================================
# 3. LINEAR REGRESSION MODEL
# ================================

# Build the multiple linear regression model
# Target variable: charges
# Explanatory variables: age, bmi, smoker, region, sex, children
model_global <- lm(charges ~ age + smoker + region + sex + children + bmi, data = data)

# Display model summary (coefficients, p-values, R², etc.)
summary(model_global)

# ================================
# 4. MODEL ASSUMPTION CHECKS
# ================================

# --------- 4.1 Normality of residuals ---------

# Histogram and QQ plot of residuals
hist(residuals(model_global), main="Histogram of Residuals")
qqnorm(residuals(model_global))
qqline(residuals(model_global))

# Shapiro-Wilk test for normality
shapiro.test(residuals(model_global))
# Note: For large datasets, this test is very sensitive

# --------- 4.2 Homoscedasticity ---------

# Residuals vs fitted values plot
plot(model_global, which = 1)

# Studentized Breusch-Pagan test
library(lmtest)
bptest(model_global)

# --------- 4.3 Multicollinearity ---------

# Variance Inflation Factor (VIF)
library(car)
vif(model_global)

# Rule of thumb:
# VIF < 2 indicates no serious multicollinearity

# --------- 4.4 Linearity ---------

# Component + Residual plots
# Used to check linear relationships between predictors and response
crPlots(model_global)

# ================================
# 5. OUTLIER DETECTION
# ================================

# --------- Studentized residuals ---------

# Identify observations with large residuals
res_stud <- rstudent(model_global)
res_stud_idx <- which(abs(res_stud) > 2)

# --------- Leverage ---------

# Identify influential observations in predictor space
lev <- hatvalues(model_global)
lev_idx <- which(lev > 2 * mean(lev))

# --------- Cook's distance ---------

# Measure overall influence of each observation
cook <- cooks.distance(model_global)
cook_idx <- which(cook > 4 / nrow(data))

# --------- Combine influential points ---------

# Points appearing in multiple detection methods
intersect_all <- Reduce(intersect, list(res_stud_idx, lev_idx, cook_idx))
intersect_all

# Points appearing in at least two criteria
all_idx <- c(res_stud_idx, lev_idx, cook_idx)
suspects <- as.numeric(names(which(table(all_idx) >= 2)))
suspects

# ================================
# 6. MODEL WITHOUT OUTLIERS
# ================================

# Refit the model after removing influential observations
model2 <- lm(charges ~ age + smoker + region + sex + children + bmi,
             data = data[-suspects, ])

# Model summary
summary(model2)

# ================================
# 7. MODEL COMPARISON
# ================================

# Compare models using AIC (lower is better)
AIC(model_global, model2)

# Compare explanatory power
summary(model2)$r.squared

# ============================================================
# END OF SCRIPT
# ============================================================
