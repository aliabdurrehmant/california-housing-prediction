# California Housing Price Prediction

Predicts median house value in California districts using regression models,
comparing a baseline approach against feature engineering and a stronger model.

## Dataset
Built-in scikit-learn California housing dataset (20,640 rows, 8 features).

## Approach
1. Explored the data with pandas (`.describe()`, `.isna()`, checked for outliers)
2. Engineered two new features: `RoomsPerHousehold`, `BedroomRatio`
3. Trained and compared two models: Linear Regression and Random Forest
4. Evaluated using MSE, MAE, and R²

## Results
| Model | R² |
|---|---|
| Linear Regression (baseline) | 0.60 |
| Linear Regression + engineered features | 0.66 |
| Random Forest + engineered features | 0.81 |

![Model comparison chart](C:\Users\aliab\Documents\PYTHON PROJECTS\PythonProject\ML project 1..png)

## Challenges I ran into
- **Isolating what actually caused an improvement** — when I changed both the
  train/test split and the engineered features at the same time between runs,
  I couldn't tell which change caused the better R² score. Had to re-run with
  a matched split (same test_size/random_state) to isolate the effect of the
  feature engineering specifically.
- **Choosing the right features to engineer** — the raw `AveRooms` and
  `AveBedrms` columns weren't very informative on their own, since they don't
  account for household size. Deriving `RoomsPerHousehold` and `BedroomRatio`
  gave the model more directly useful signal, improving R² from 0.60 to 0.66
  even before switching models.
- **Understanding evaluation metrics beyond just accuracy** — comparing MSE,
  MAE, and R² together (rather than picking one) gave a clearer picture of
  model performance, since each highlights something different (MSE punishes
  large errors more heavily, MAE is easier to interpret directly).

## Notes
This was my first time using matplotlib — the chart above was built while
learning the basics. The pandas/scikit-learn work reflects more practice.

## What I'd improve next
- Hyperparameter tuning on the Random Forest (`n_estimators`, `max_depth`)
- Feature scaling for models sensitive to it
- Try additional models (e.g. Gradient Boosting)
- More visualizations (e.g. predicted vs actual scatter plot)