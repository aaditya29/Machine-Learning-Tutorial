# Important Notes Of Week 2

## Multiple Linear Regression

**What is Multiple Linear Regression (MLR)?**<br>

- **Extension of Simple Regression:** Multiple linear regression is an extension of simple linear regression, where we had just one independent variable to predict our dependent variable. With MLR, we introduce multiple independent variables to explain the behavior of our dependent variable.
- **The Formula:** The core idea is represented by this equation:

  `Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn + error`
  Where:

  - Y is the dependent variable (what we want to predict)
  - X1, X2, ..., Xn are the independent variables (predictors)
  - b0 is the intercept (the expected value of Y when all X's are zero)
  - b1, b2, ..., bn are the coefficients (they tell us how much a one-unit change in an independent variable affects the dependent variable)
  - error is the residual or unexplained variation

**Why Use Multiple Linear Regression?**<br>

- **Greater Predictive Power:** In most cases, a single factor can't fully explain the variation in a real-world phenomenon. Using multiple independent variables enhances your model's ability to understand and predict your dependent variable.
- **Unraveling Complex Relationships:** MLR helps you figure out which independent variables have the strongest impact on your outcome and whether certain variables interact with each other.
