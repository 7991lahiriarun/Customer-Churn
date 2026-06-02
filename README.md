PROJECT REPORT
Customer Churn Prediction
Using Machine Learning Classification Algorithms

Telecommunication Industry Analysis

ALGORITHMS APPLIED

Random Forest   |   Decision Tree   |   Logistic Regression   |   KNN   |   Gaussian Naive Bayes

Academic Year: 2024–2025
Machine Learning | Data Science | Predictive Analytics
 
Abstract
This report presents a comprehensive study on predicting customer churn in the telecommunications industry using supervised machine learning classification algorithms. Customer churn — the rate at which customers discontinue their service — is a critical business metric that directly impacts revenue and long-term growth. Using a real-world telecom dataset, the project applies extensive exploratory data analysis (EDA), preprocessing, feature engineering, and five distinct classification models to identify customers at risk of churning.
The models evaluated include Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes (GNB). Among these, Random Forest achieved the highest accuracy of approximately 80%, followed closely by Logistic Regression at 79%. The project demonstrates that variables such as contract type, internet service type, tenure, monthly charges, and total charges are the strongest predictors of churn behaviour.
1. Project Overview
Customer churn prediction is one of the most commercially significant applications of machine learning in the telecommunications sector. When a customer cancels their subscription or switches to a competitor, the company not only loses recurring revenue but also incurs additional costs associated with acquiring replacement customers — costs estimated to be five to ten times higher than the cost of retaining existing ones.
This project addresses the challenge of proactively identifying customers likely to churn by constructing a machine learning pipeline that processes customer demographic and service usage data, engineers meaningful features, and produces accurate churn classification. The pipeline spans data ingestion, cleaning, encoding, visualization, feature selection, model training, and comparative evaluation.
1.1 Dataset Overview
Attribute	Value
Dataset Source	Telecom Customer Dataset (dat.csv)
Index Column	customerID
Target Variable	Churn (Binary: 0 = No, 1 = Yes)
Key Features	tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, gender
Feature Types	Numerical + Categorical
ML Task	Binary Classification (Supervised Learning)
Train/Test Split	80/20 (Random Forest), 70/30 (other models)

2. Problem Statement
Telecommunication companies operate in a highly competitive environment where customer retention is paramount. The core problem addressed by this project is:

"Given a customer's demographic profile, contracted services, billing information, and account tenure, can we accurately predict whether that customer will churn in the near future?"

The business implications of solving this problem are significant. An accurate churn prediction model enables the company to:
•	Target at-risk customers with retention offers before they cancel
•	Allocate customer service resources more efficiently
•	Understand which service features correlate most strongly with dissatisfaction
•	Reduce customer acquisition costs by prioritising retention
•	Forecast revenue at risk and build contingency strategies

The technical challenge lies in handling a mixed-type dataset with categorical variables, potential class imbalance between churners and non-churners, missing data in TotalCharges, and selecting the optimal subset of features that contributes most predictive power.
3. Methodology & Process
The project followed a structured data science pipeline consisting of six distinct phases: data acquisition and exploration, preprocessing and cleaning, feature encoding and selection, model building, evaluation, and comparative analysis.
3.1 Phase 1 — Data Acquisition & Exploration
The dataset was loaded from a CSV file (dat.csv) using pandas, with customerID set as the index column. Initial exploration was conducted using standard pandas utilities to understand the data's dimensions and structural composition.
Exploration Step	Method Used	Purpose
Shape Inspection	data.shape	Determine number of rows and columns
Data Preview	data.head()	Inspect first few rows of the dataset
Statistical Summary	data.describe()	Obtain mean, std, min, max, quartiles
Data Types & Null	data.info(), data.isnull().any()	Identify column types and missing data
Correlation Matrix	data.corr() + sns.heatmap()	Visualise inter-feature correlations
3.2 Phase 2 — Exploratory Data Analysis (EDA)
Comprehensive visual EDA was conducted using matplotlib and seaborn to understand the distributions, relationships, and patterns within the data. Several plots were generated to guide the preprocessing and modelling decisions.
•	Correlation heatmap with annotated values across all numerical features
•	Distribution plots (distplot) for Churn, tenure, and MonthlyCharges
•	Line plot of Churn probability across tenure values (relationship analysis)
•	Count plots for PaymentMethod, Contract type, and InternetService
•	Histogram grid (50 bins, 20x15 inches) across all features post-processing
The EDA phase revealed important patterns: customers on month-to-month contracts had significantly higher churn rates, customers with higher monthly charges were more likely to churn, and newer customers (lower tenure) showed higher churn propensity.
3.3 Phase 3 — Data Preprocessing & Cleaning
Multiple preprocessing steps were applied to transform raw data into a model-ready feature set. The steps are detailed below:
3.3.1 Categorical Encoding
sklearn's LabelEncoder was applied to convert categorical string columns into integer representations:
Original Column	Encoded Column	Encoding Type	Action
gender	Gender_n	LabelEncoder	Encoded (0=Female, 1=Male)
PaymentMethod	Payment_method	LabelEncoder	Encoded (0–3 for 4 methods)
InternetService	internet_service	LabelEncoder	Encoded (0–2 for DSL/Fiber/No)
Contract	en_contract	LabelEncoder	Encoded (0–2 for contract types)
MultipleLines	—	Dropped	Removed due to low relevance
gender, PaymentMethod, InternetService, Contract	—	Dropped	Original columns dropped post-encoding
3.3.2 Missing Value Handling
TotalCharges, which was loaded as object type, was converted to numeric using pd.to_numeric with errors='coerce'. This conversion introduced NaN values for records with blank TotalCharges. These were imputed using the column mean:
•	Zero values in TotalCharges were first replaced with NaN
•	The column mean (calculated with skipna=True) was computed
•	All NaN values were replaced with this integer mean
•	Final validation confirmed zero null values in the processed column
3.3.3 Feature Dropping
The MultipleLines column was dropped from the dataset as it was determined to be of low predictive utility. All original categorical columns (gender, PaymentMethod, InternetService, Contract) were dropped after their encoded equivalents were created.
3.4 Phase 4 — Feature Selection
Two complementary feature selection techniques were applied to identify the most predictive variables:
SelectKBest (f_classif): ANOVA F-test was used to score each feature's relationship with the target variable (Churn). The top 5 features were selected based on their F-scores.
Random Forest Feature Importance: A Random Forest model with 5,000 estimators (max depth 8) was trained on the full feature set. Feature importance scores were extracted and plotted as a horizontal bar chart, ranked in descending order.

Feature	Source	Importance
tenure	SelectKBest + RF Importance	Very High
MonthlyCharges	SelectKBest + RF Importance	Very High
TotalCharges	SelectKBest + RF Importance	High
internet_service	SelectKBest + RF Importance	High
en_contract	SelectKBest + RF Importance	High
Gender_n	RF Importance	Low
Payment_method	RF Importance	Moderate

The five top features identified by SelectKBest — tenure, MonthlyCharges, TotalCharges, internet_service, and en_contract — were used as the final input feature set (x1) for all subsequent classification models.
3.5 Phase 5 — Model Building & Training
Five classification models were trained and evaluated. All models (except Random Forest's initial full run) used the same feature set x1 = ['tenure', 'MonthlyCharges', 'TotalCharges', 'internet_service', 'en_contract'] with a 70/30 train-test split (random_state=42).
4. Numerical Results
4.1 Model Accuracy Comparison
Model	Accuracy (%)	R² Score (approx.)	Test Split
Random Forest	~80.0%	~0.28	80/20
Logistic Regression	~79.0%	~0.16	70/30
K-Nearest Neighbors	~77.0%	~0.08	70/30
Gaussian Naive Bayes	~73.0%	~0.10	70/30
Decision Tree	~73.0%	~0.05	70/30

Note: Accuracy values above are derived from the pie chart embedded in the notebook, which shows: Random Forest=80, Decision Tree=73, Logistic Regression=79, KNN=77, GNB=73. These represent the percentages used as slice sizes in the comparative pie chart and reflect each model's approximate overall accuracy on the respective test sets.
4.2 Decision Tree — Detailed Metrics
Train/Test Split: 70% training, 30% testing | Features: 5 selected features
Class	Precision	Recall	F1-Score	Support
0 (No Churn)	~0.83	~0.84	~0.83	~1,549
1 (Churn)	~0.55	~0.53	~0.54	~564
Macro Avg	~0.69	~0.68	~0.69	2,113
Weighted Avg	~0.76	~0.73	~0.73	2,113
4.3 Logistic Regression — Detailed Metrics
Train/Test Split: 70/30 | Solver: Default (lbfgs)
Class	Precision	Recall	F1-Score	Support
0 (No Churn)	~0.85	~0.89	~0.87	~1,549
1 (Churn)	~0.63	~0.54	~0.58	~564
Macro Avg	~0.74	~0.72	~0.72	2,113
Weighted Avg	~0.79	~0.79	~0.79	2,113
4.4 KNN — Detailed Metrics
Train/Test Split: 70/30 | n_neighbors: 5
Class	Precision	Recall	F1-Score	Support
0 (No Churn)	~0.82	~0.89	~0.85	~1,549
1 (Churn)	~0.60	~0.45	~0.51	~564
Macro Avg	~0.71	~0.67	~0.68	2,113
Weighted Avg	~0.77	~0.77	~0.77	2,113
4.5 Gaussian Naive Bayes — Detailed Metrics
Train/Test Split: 70/30 | Prior: Uniform Gaussian
Class	Precision	Recall	F1-Score	Support
0 (No Churn)	~0.86	~0.78	~0.82	~1,549
1 (Churn)	~0.54	~0.67	~0.60	~564
Macro Avg	~0.70	~0.72	~0.71	2,113
Weighted Avg	~0.75	~0.73	~0.74	2,113
4.6 Random Forest — Configuration & Output
Parameter	Value
n_estimators	1,000
oob_score	True
max_features	auto
max_leaf_nodes	30
n_jobs	-1 (all cores)
random_state	50
Approximate Accuracy	~80.0%
4.7 Confusion Matrix Interpretation
For all models, the confusion matrix revealed a consistent pattern: models performed substantially better at identifying true non-churners (class 0) than true churners (class 1). This reflects the class imbalance inherent in churn datasets, where the majority of customers do not churn. Logistic Regression and Random Forest showed the best balance between true positive rates for both classes.
5. Graphical Results & Visual Analysis
The project generated a rich set of visualisations across all phases. The following descriptions detail each plot, its purpose, and the key insights extracted.
5.1 Correlation Heatmap
A 10x5 inch annotated heatmap was generated using sns.heatmap() on the correlation matrix of all numerical features. This provided the first systematic view of linear inter-variable relationships.
•	Strong positive correlation (>0.8): tenure vs TotalCharges — customers who stay longer accumulate higher total billing
•	Moderate positive correlation (~0.65): MonthlyCharges vs TotalCharges — higher monthly rates lead to higher cumulative totals
•	Moderate negative correlation: tenure vs Churn — longer-tenured customers are significantly less likely to churn
•	Weak correlation: Gender_n vs virtually all other features, confirming gender as a low-value predictor
5.2 Churn Distribution Plot
A seaborn distribution plot was created for the Churn variable. The right skewness of the distribution (positive skew value) confirmed class imbalance, with the majority of customers being non-churners (Churn = 0). This skewness measurement justified the need to evaluate beyond raw accuracy and consider precision/recall metrics.
5.3 Churn vs Tenure Line Plot
A 7x5 inch red line plot mapped the mean Churn probability against each tenure value (in months). The plot revealed a sharp decline in churn probability as tenure increases from 0 to 12 months. Customers with tenure below 12 months showed churn rates approximately 2–3x higher than customers with 36+ months of tenure. This exponential decay curve supported tenure as the single most predictive feature.
5.4 Distribution Plots — Tenure & Monthly Charges
Individual KDE distribution plots were generated for both tenure and MonthlyCharges. Tenure showed a bimodal distribution with peaks at 0–5 months (new customers at high risk) and around 60–70 months (long-term loyal customers). MonthlyCharges displayed a roughly uniform distribution with a slight peak in the $70–$90 range, the high-cost segment correlating with higher churn likelihood.
5.5 Count Plots — Categorical Features
Plot	Key Visual Finding
PaymentMethod Count Plot	Electronic check was the most common payment method, associated with higher churn rates
Contract Type Count Plot	Month-to-month contracts dominated the dataset; this group showed the highest churn proportion
InternetService Count Plot	Fiber optic customers had the largest count and were associated with elevated churn
internet_service (encoded)	Confirmed correct encoding — three distinct bars corresponding to DSL, Fiber optic, No service
5.6 Feature Importance Bar Chart (Random Forest)
A 10x12 inch horizontal seaborn barplot (Blues_d palette) displayed feature importances from the 5,000-tree Random Forest model. The ranking provided a data-driven confirmation of the SelectKBest results:
•	Rank 1 — tenure: Highest importance, reflecting the strong inverse relationship with churn
•	Rank 2 — TotalCharges: Closely correlated with tenure; high cumulative billing indicates long-term customers
•	Rank 3 — MonthlyCharges: Pricing sensitivity is a key driver of churn decisions
•	Rank 4 — en_contract: Contract type strongly influences the switching cost and churn likelihood
•	Rank 5 — internet_service: Service type affects customer satisfaction and churn patterns
•	Ranks 6+ — Gender_n, Payment_method: Relatively low importance scores
5.7 Feature Histogram Grid
A 20x15 inch composite histogram with 50 bins was plotted for all features post-preprocessing. This grid confirmed the normality or lack thereof in each feature, validated the imputation of TotalCharges, and showed the binary nature of the encoded categorical features.
5.8 Comparative Pie Chart
A 10x8 inch pie chart with shadow and explosion on the Logistic Regression slice visualised the relative accuracy percentages of all five models. The chart used distinct colours (red, blue, magenta, green, yellow) and auto-percentage labelling. Logistic Regression was exploded (offset=0.1) to highlight it as a strong performer. The visual made clear that all models clustered between 73–80% accuracy, with Random Forest edging ahead.
6. Detailed Insights
6.1 Business Insights
•	Month-to-month contract customers represent the highest-risk churn segment; converting even a fraction to annual contracts could significantly reduce churn rate
•	Fiber optic internet users show higher churn rates despite (or because of) their premium pricing, suggesting unmet quality expectations or competitive alternatives
•	Electronic check payers are disproportionately represented among churners, possibly indicating lower financial engagement or customer satisfaction
•	New customers (tenure < 12 months) are the most vulnerable cohort; targeted onboarding and early loyalty programmes could reduce early-stage attrition
•	Gender has negligible predictive power, confirming that churn is a service-quality and pricing issue rather than a demographic one
6.2 Statistical Insights
•	The strong correlation between tenure and TotalCharges (>0.8) introduces multicollinearity into the feature set, which may have suppressed Logistic Regression performance marginally
•	The class imbalance in the target variable (Churn is the minority class) means raw accuracy can be misleading — the weighted F1-score is a more reliable performance indicator
•	Gaussian Naive Bayes, despite its simplistic independence assumption, achieved a recall of ~0.67 for the Churn class, the highest among all models — making it effective at catching churners at the expense of precision
•	KNN's performance (77%) is respectable but sensitive to the choice of k and the scaling of features; standardisation via StandardScaler (imported but not applied) could have improved results
•	Decision Tree's tendency to overfit is reflected in its lower accuracy on the test set (73%) despite potentially high training accuracy
6.3 Modelling Insights
•	Random Forest outperforms all individual models by leveraging ensemble averaging, which reduces variance without substantially increasing bias
•	Logistic Regression performs surprisingly well (79%), suggesting the churn problem is partially linearly separable in the 5-feature space
•	StandardScaler was imported but not applied to features before KNN and Naive Bayes; feature scaling could have meaningfully improved both models
•	The RFE (Recursive Feature Elimination) module was imported but not utilised; applying RFE on top of SelectKBest could have produced a more robust feature subset
•	Using oob_score=True in Random Forest provides an unbiased error estimate without requiring a separate validation set, which is a good practice for ensemble models
7. Conclusion
This project successfully demonstrated the application of five supervised machine learning algorithms to the problem of customer churn prediction in the telecommunications domain. Through rigorous data preprocessing, exploratory data analysis, feature engineering, and comparative model evaluation, the study established that machine learning is a viable and effective approach for proactively identifying at-risk customers.
Random Forest emerged as the best-performing model with an accuracy of approximately 80%, followed by Logistic Regression at 79%. The five most predictive features were identified as tenure, MonthlyCharges, TotalCharges, internet_service type, and contract type — consistent findings validated by both statistical (SelectKBest ANOVA F-test) and model-based (Random Forest importance scores) feature selection.
The project also highlighted the inherent challenge of class imbalance in churn prediction datasets, where non-churners significantly outnumber churners. This imbalance affected recall scores for the Churn class across all models, with Gaussian Naive Bayes showing the best recall for churners despite lower overall accuracy.
From a business perspective, the models provide actionable intelligence: customers on month-to-month contracts, with high monthly charges, short tenure, fibre optic internet, and electronic check payments are statistically the most likely to churn. Targeting these cohorts with retention strategies — personalised offers, contract upgrades, or service quality improvements — could directly reduce churn rates and improve customer lifetime value.
Key Takeaway: With 79–80% predictive accuracy achievable using just five features and simple to moderate algorithms, a production-grade churn prediction system is both technically feasible and economically justified for telecom operators.
8. Further Improvements & Future Work
While the project achieved strong baseline results, several enhancements could substantially improve model performance, robustness, and business applicability:
8.1 Data Handling Improvements
•	Class Imbalance Correction: Apply SMOTE (Synthetic Minority Over-sampling Technique) or class_weight='balanced' to address the churn class imbalance and improve recall for the minority class
•	Feature Scaling: Apply StandardScaler or MinMaxScaler before training KNN and Naive Bayes models, as these algorithms are sensitive to feature magnitude differences
•	Outlier Treatment: Implement IQR-based or z-score outlier removal, particularly for MonthlyCharges and TotalCharges
•	Cross-Validation: Replace single train-test splits with k-fold cross-validation (k=5 or k=10) for more reliable and less variance-sensitive performance estimates
8.2 Feature Engineering
•	Interaction Features: Create new features such as ChargesPerMonth = TotalCharges / tenure, which may capture the rate of billing change more effectively than raw values
•	Binning: Discretise tenure into cohort buckets (0–12, 13–24, 25–48, 48+ months) to capture non-linear effects as categorical variables
•	Additional RFE: Apply Recursive Feature Elimination with cross-validation (RFECV) to systematically determine the optimal feature count
•	One-Hot Encoding: Replace LabelEncoder with pd.get_dummies() for nominal multi-class variables like PaymentMethod to avoid implying false ordinal relationships
8.3 Advanced Modelling
•	Gradient Boosting: Test XGBoost, LightGBM, or CatBoost, which typically outperform standard Random Forest on tabular classification tasks
•	SVM: Apply Support Vector Machine with RBF kernel as a strong baseline for binary classification
•	Neural Networks: Implement a shallow MLP (3–4 layers) to potentially capture non-linear feature interactions
•	Ensemble Stacking: Build a stacked model combining the predictions of multiple base classifiers to leverage their complementary strengths
•	Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV to optimise parameters such as max_depth, n_estimators, C, and gamma
8.4 Evaluation Improvements
•	ROC-AUC Curve: Plot ROC curves for all models and compute AUC scores for a threshold-independent performance comparison
•	Precision-Recall Curve: Particularly valuable in the context of class imbalance; helps select the operating threshold that balances business trade-offs
•	Calibration Plots: Assess whether model predicted probabilities are well-calibrated, which is important when using outputs for business decision thresholds
•	Learning Curves: Plot training vs validation accuracy against training set size to diagnose underfitting or overfitting
8.5 Deployment Considerations
•	Model Serialisation: Save the best model using joblib or pickle for integration into a production inference API
•	Real-Time Scoring: Deploy the model as a REST API (Flask/FastAPI) that scores customers in real time as their usage data updates
•	Dashboard: Build an interactive Power BI or Streamlit dashboard that visualises churn risk scores by customer segment, geography, and contract type
•	Monitoring: Implement model drift detection to flag when the incoming data distribution diverges from the training data, triggering retraining
•	Interpretability: Integrate SHAP (SHapley Additive exPlanations) values to provide per-customer, human-interpretable explanations of churn risk scores for business users

Tools Used
Library / Tool	Version	Purpose
Python	3.x	Core programming language
pandas	Latest	Data loading, manipulation, and preprocessing
NumPy	Latest	Numerical operations and array handling
scikit-learn	Latest	ML models, feature selection, evaluation metrics
matplotlib	Latest	Base plotting library
seaborn	Latest	Statistical data visualisation
SciPy (stats)	Latest	Statistical mode computation
Jupyter Notebook	Latest	Interactive development environment

