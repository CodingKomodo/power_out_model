# power_out_model


# power_out_model

### Prediction Problem and Type

**Prediction Problem:** Predicting the duration of power outages given data from the Purdue Outage dataset.  
**Type:** Regression

### Response Variable

The response variable is the duration of power outages, which represents the length of time a power outage persists. This variable was chosen because it is continuous quantitative data, and the goal is to predict the specific duration of outages.

### Features for Prediction

The features used for prediction should be the ones available and relevant at the time of predicting the outage duration. Features like geographical location, weather conditions, outage start time, utility company information, historical outage data, and any other relevant parameters available up to the point of making predictions should be considered. These might include:

1. Geographical location (State, City)
2. Weather conditions (Temperature, Precipitation, Storm Severity)
3. Time of day or year
4. Utility company information (Service area, past outage records)

### Evaluation Metric

The choice of evaluation metric should align with the nature of the regression problem. Common regression metrics include:


1. Root Mean Squared Error (RMSE)
2. R-squared (Coefficient of determination)

The selection among these metrics should depend on the specific characteristics and requirements of the problem. For instance:

- **RMSE** RMSE is sensitive to outliers due to squaring differences, but it provides a better indication of how far off predictions are on average.
- **R-squared** measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It indicates the goodness of fit of the model.

The choice among these metrics depends on the preference for handling different aspects of prediction accuracy (e.g., sensitivity to outliers, emphasis on larger errors, or explaining variance).

### Justification

For predicting outage durations, a regression model is appropriate as it deals with continuous quantitative data. RMSE, and R-squared are common metrics used for regression problems, each providing insights into different aspects of prediction accuracy. The choice of these metrics should consider factors like interpretability, sensitivity to outliers, and the specific goals of the prediction task.


Note: Make sure to justify what information you would know at the “time of prediction” and to only train your model using those features. For instance, if we wanted to predict your final exam grade, we couldn’t use your Project 5 grade, because Project 5 is only due after the final exam!



## Base Model Description

**Type of Model:** Linear Regression

### Features in the Model

The selected features used in the model are:
- `DEMAND.LOSS.MW`: Quantitative feature representing the amount of demand loss in megawatts.
- `CUSTOMERS.AFFECTED`: Quantitative feature indicating the number of customers affected by the outage.
- `RES.PRICE`, `COM.PRICE`, `IND.PRICE`: Quanatative features representing residential, commercial, and industrial electricity prices.

**Quantitative Features:** `DEMAND.LOSS.MW`, `CUSTOMERS.AFFECTED`, `RES.PRICE`, `COM.PRICE`, `IND.PRICE`
**Nominal Features:** `POSTAL.CODE`

**Ordinal Features:** NONE

### Data Preprocessing

1. **Handling Missing Values:** We dropped all the missing columns
### Model Performance

Our R^2 was around 0.23 while our RMSE was 
<!-- The performance of the model is evaluated using Mean Squared Error (MSE) as a metric. However, the code provided doesn't calculate or report the performance metrics like MSE. To evaluate the model's performance, it's necessary to compute metrics like MSE on the test set (`y_test` and `y_pred`). -->

### Model Evaluation

<!-- To assess the model's "goodness," we need to compare its performance metrics (such as MSE) against a baseline or other models. A "good" model should demonstrate lower MSE, indicating better accuracy in predicting outage durations. -->

### Improvements and Considerations

1. **Feature Engineering:** Consider incorporating more features that might influence outage durations, such as weather conditions, time of day/year, historical outage data, or geographical factors.
  
2. **Feature Encoding:** Nominal features like `RES.PRICE`, `COM.PRICE`, `IND.PRICE` might benefit from proper encoding techniques like one-hot encoding to translate them into numerical form before training the model.

3. **Model Selection:** Experiment with other regression models (e.g., Decision Trees, Random Forests, Gradient Boosting) and evaluate their performance against the Linear Regression model.

4. **Hyperparameter Tuning:** Perform grid search or other hyperparameter tuning techniques to optimize the model's performance.

Without performance metrics reported, it's challenging to definitively assess the model's "goodness." However, further improvements and evaluations as suggested can enhance the model's predictive power for outage durations.


