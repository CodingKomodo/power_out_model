___
### Prediction Problem and Type:

The prediction problem is **multiclass classification**. We are predicting the `CAUSE.CATEGORY` variable, which represents the categorized cause of an outage event. This variable aims to classify the cause category (e.g., equipment failure, human error, etc.) based on given features.

### Response Variable Justification:

- **`CAUSE.CATEGORY`:** This variable is chosen as the response variable because understanding the cause of an outage event is crucial for diagnosing issues, implementing preventive measures, and optimizing resource allocation in maintaining the system's reliability. It provides actionable insights for resolving and preventing such incidents.

### Evaluation Metric:

The chosen evaluation metric is **accuracy_score** for assessing model performance. 

**Justification for Accuracy Score:**
- **Advantages:** Accuracy provides a straightforward interpretation - the proportion of correctly classified instances among the total instances. For multiclass classification, it gives an overall performance view.
- **Suitability:** In many scenarios where the classes are balanced, accuracy can be a reliable metric. It's easy to understand and interpret.

### Rationale for Metric Selection:

- **Accuracy vs. F1-Score:** Accuracy is suitable when classes are balanced and each class is of equal importance. However, in cases of class imbalance, where different misclassification costs exist, or when false positives/negatives have significantly different implications, F1-score might be preferred. In this scenario, if the classes are relatively balanced and each cause category's importance is equal, accuracy could be a reasonable choice. However, it's essential to consider other metrics like precision, recall, and F1-score for a more comprehensive understanding of the model's performance, especially if class imbalance or varying misclassification costs are present.

The objective here is to predict the cause category of outages based on the provided features using a multiclass classification approach, and the chosen metrics align with the nature of this predictive problem, considering the known features at the time of prediction and the significance of correctly identifying the cause categories for mitigation and resolution of outages.


### Base Model:

1. **Type of Model:** RandomForestClassifier - This model is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees.
  
2. **Features in the Model:**

   a. **Quantitative Features:**
      - `N/A`
   
   b. **Ordinal Features:**
      - `ANOMALY.LEVEL`
      - `MONTH`
   
   c. **Nominal Features:**
      - `POSTAL.CODE`

  
### Encoding and Feature Transformation:

1. **Encoding:**
   - One-Hot Encoding was applied to the 'POSTAL.CODE' feature using `OneHotEncoder(handle_unknown='ignore')` within a `ColumnTransformer` to convert categorical data into a format suitable for machine learning models.

2. **Feature Transformation:**
   - No explicit feature transformation or scaling was applied to the other features, indicating they might have been considered in their original form during the modeling process.

### Performance Evaluation:

- **Interpretation of Performance:**
  - By using `y_test` and `prediction` obtained from `pl.predict(X_test)`, we were able to find an accuracy score of around 0.63.
  - The accuracy score for a simple model, where we predict the greatest amount of one of cateogry which is `severe weather`, we would get an accuracy score of around 53%.
  - Considering the accuracy score for a simple model, an accuracy score of 63% is a decent starting point and predicts better than a simple model
  
- **Considerations for Model Improvement:**
  - Adding more features/columns such as `OUTAGE.DURATION`, `MONTH`,`CUSTOMERS.AFFECTED`.
  - Tune hyperparameters using techniques like GridSearchCV to improve model performance.
  - Feature engineering: Extract more informative features or transform existing ones.
  

Without the specific accuracy score or additional metrics, it's challenging to definitively categorize the model as "good." An "accurate" model highly depends on the problem's context, the data quality, feature selection, and domain understanding.

Consideration of these aspects will provide a clearer perspective on the model's effectiveness and guide improvements to achieve a better-performing model.


### Features Added and Their Relevance:

1. **'MONTH' Feature:** 
   - **Rationale:** The 'MONTH' feature might capture seasonal variations or trends in outage occurrences. Certain months might have higher or lower outage frequencies due to weather patterns, maintenance schedules, or increased usage, providing valuable predictive patterns.

2. **'CUSTOMERS.AFFECTED' Feature:**
   - **Rationale:** This feature likely represents the number of customers affected by the outage. It could play a significant role in understanding the severity of an outage event and might correlate with specific cause categories. For instance, equipment failure might affect more customers than human error, aiding in the prediction task by providing insights into the outage's impact.

### Modeling Algorithm and Hyperparameter Tuning:

1. **Modeling Algorithm:** RandomForestClassifier
   - **Rationale:** Random Forests handle nonlinear relationships well, are resistant to overfitting, and perform implicit feature selection. These properties make them suitable for complex classification tasks and handling diverse feature sets.

2. **Hyperparameters and Tuning:**
   - **Hyperparameter:** `max_depth` of the Random Forest
   - **Tuning Method:** Utilized GridSearchCV to explore different `max_depth` values.
   - **Best Performing Hyperparameter:** `max_depth: 22`

### Model Performance Improvement over Baseline:

- **Baseline Model:** 
  - Features: 'POSTAL.CODE', 'ANOMALY.LEVEL', 'OUTAGE.DURATION'
  - Accuracy: Not specified

- **Final Model:**
  - Added Features: 'MONTH', 'CUSTOMERS.AFFECTED'
  - Hyperparameter Tuning: `max_depth: 22`
  - Accuracy: 0.896 (89.6%)

### Improvement Explanation:

- **Feature Addition:** The inclusion of 'MONTH' and 'CUSTOMERS.AFFECTED' features enriches the model by providing more diverse information regarding seasonal patterns and the impact of outages on customers. This additional information allows the model to capture nuanced relationships between these features and the cause categories, leading to improved predictive performance.

- **Hyperparameter Tuning:** The optimization of the `max_depth` hyperparameter via GridSearchCV likely helped the model achieve better generalization by preventing overfitting and improving its ability to capture complex relationships within the data.

- **Performance Comparison:** The accuracy of 89.6% in the final model represents a significant improvement over the baseline model (accuracy not specified). This enhancement suggests that the added features and optimized hyperparameters contributed positively to the model's predictive capabilities, providing a more accurate classification of outage causes.

### Fairness Analysis

<iframe src="confusion_matrix.html" width=800 height=600 frameBorder=0></iframe> 

- **Precisioin:** 0.8741150264196992
- **Accuracy:** 0.9009433962264151



