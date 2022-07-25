# Insurance All Cross-Sell

### Cross-sell propensity score list built with a Classification Model

<img src="image/cover_insurance_all.png" width="1000">

## 1. Abstract:

**Disclaimer:** Insurance All is a fictitious company, according to the context presented in this project.

This **Data Science Project** was inspired by this [kaggle Challenge](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) and presents the development of a Classification Machine Learning Model, more specifically a Learning to Rank Model, used to generate a propensity score to purchase a new product for a company's customer list. 

In the commercial area, this sales strategy is known as Cross-Sell and can be defined as a sales technique that involves selling an additional product or service to an existing customer [(Wikipedia)](https://en.wikipedia.org/wiki/cross-sell).

The list with the purchase propensity score was the solution found for a business limitation: from a dataset with 127 thousand customers, the sales team would be able to contact 20 thousand people during the campaign period. And compared to a random selection of customers to be contacted, the machine learning model developed proved to be about 3 times more efficient, generating an extra gain of 25 million dollars.

**At the end of the project, two data products were presented to the commercial team:**

* 1) The [ordered list](https://docs.google.com/spreadsheets/d/1vNiaBNN6GXCN-k3ZkEtUqUoJ2NzDeIdRT8PwRNMxO1c/edit?usp=sharing) of the 127 thousand customers classified by the highest purchase propensity;
* 2) A [script](https://github.com/vitorhmf/cross-sell/blob/main/google_sheet_script/InsuranceAll.gs) to be put into Google Sheets that allows access to the trained model, put into production on Heroku Cloud. With this spreadsheet, as shown in the example below, the commercial team can easily perform simulations and queries on the purchase propensity of a specific group of customers.


<img src="image/propensity_score_simulation.gif" width="800">



**Keywords:** Python, Pandas, Numpy, Seaborn, Matplotlib, Pickle, Scikit Learn, Classification Model, Random Forest, XGBoost, KNN, LGBM, Flask, Heroku

## 2. Methodology

The CRISP-DM methodology was the guide for this data science project development. 

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an industry-proven way to guide your data mining efforts and it includes descriptions of the typical phases of a project, the tasks involved with each phase, and an explanation of the relationships between these tasks.

<img src="image/crisp_process.jpg" width="500">

**Source:* [IBM Docs](https://www.ibm.com/docs/en/spss-modeler/18.2.0?topic=dm-crisp-help-overview)

To direct your reading, below are links to the development carried out at each stage of the CRISP cycle:

* [Business Understanding](https://github.com/vitorhmf/cross-sell#3-business-understanding)
* [Data Understanding](https://github.com/vitorhmf/cross-sell#4-data-understanding)
* [Data Preparation](https://github.com/vitorhmf/cross-sell#5-data-preparation)
* [Machine Learning Modeling](https://github.com/vitorhmf/cross-sell#6-machine-learning-modeling)
* [Evaluation](https://github.com/vitorhmf/cross-sell#7-evaluation)
* [Depoyment](https://github.com/vitorhmf/cross-sell#8-deployment)

## 3. Business Understanding

### 3.1. Context

The Insurance All is a company that works with health insurance for its customers and now the product team is analyzing the possibility of offering a new product to its customers: auto insurance.

In this case, we worked with two datasets, both composed of Insurance All customers who already have the company's health insurance.

* In the first dataset, we have the result of a survey carried out with 381,109 customers. This result was saved in the database along with other customer attributes.
* In the second dataset, we have the attributes of another 127,037 customers, who did not respond to the survey. These customers will be offered the new auto insurance product.

Considering that the sales team has the capacity to make 20,000 calls within the campaign period, we need to answer the business team what percentage of customers interested in purchasing auto insurance, the sales team will be able to contact by making the 20,000 calls.

And if the sales team's capacity increases to 40,000 calls, what percentage of customers interested in purchasing auto insurance will the sales team be able to contact?

| Feature                | Definition                                                                                               |
|------------------------|----------------------------------------------------------------------------------------------------------|
| id                     | Unique ID for the customer                                                                               |
| gender                 | Gender of the customer                                                                                   |
| age                    | Age of the customer                                                                                      |
| driving_license        | 0 : Customer does not have DL, 1 : Customer already has DL                                               |
| region_code            | Unique code for the region of the customer                                                               |
| previously_insured     | 1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance                  |
| vehicle_age            | Age of the Vehicle                                                                                       |
| vehicle_damage         | 1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past. |
| anual_pemium           | The amount customer needs to pay as premium in the year                                                  |
| policysaleschannel     | Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc. |
| vintage                | Number of Days, Customer has been associated with the company                                            |
| response               | 1 : Customer is interested, 0 : Customer is not interested                                               |

*Source:* [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)


### 3.2. Business assumption: 

* For educational purposes only, the dollar has been set as the default currency for the problem.
 
## 4. Data Understanding

### 4.1. Data Cleaning

To build an overview of the data, the following steps were performed:
* Change the columns name to sneak_case;
* Shows the data dimensions (rows and columns);
* Check and Fillout NA: the dataset did not have any missing values;


### 4.2 Data Descriptive: 

A quick descriptive analysis of numerical and categorical variables was performed.
 
**Numerical Attributes:**

<img src="image/num_attributes.png" width="600">

**Categorical Attributes:**

<img src="image/cat_attributes.png" width="800">
 
### 4.3. Feature Engineering

Before performing the feature engineering, a mental map was created to evaluate the relationship between the sales phenomenon and the agents that act on it, as well as the attributes of each agent.

<img src="image/mind_map.png" width="800">

From this mental map, business hypotheses were created in order to develop the understanding of the case and raise new variables that are important to derive from the original dataset for the creation of the machine learning model.
 
In this step, the following features were created:
* Vehicle Age: changed to snake case;
* Vehicle Damage: changed to 1 (yes) or 0 (no);

### 4.4. Data Filtering

In this project, it was not necessary to apply any type of filter to the dataset

### 4.5. Exploratory Data Analysis

In the data exploration, univariate, bivariate and multivariate analyzes were performed. Of the business insights obtained in this phase, 3 stood out for presenting different results than expected:

**Younger drivers aren't more likely to buy vehicle insurance.** Just 6,1% of the young group (20-35) are likely to buy, while 20,9% of the adults group (35-55) and 11,8% of the seniors (55+) group are likely to buy.

<img src="image/h2.png" width="600">

**Customers with older cars are more likely to buy:** Just 4% of customers with cars less than one year old were interested in insurance.

<img src="image/h7.png" width="600">

**Customers that already had vehicle damage are more likely to buy:** The surprise here was that practically no one showed interest in the group of those who had no previous accidents.

<img src="image/h8.png" width="600">

[Complete Notebook](https://github.com/vitorhmf/cross-sell/blob/main/notebooks/v07_cross_sell_review2.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)
 
## 5. Data Preparation

### 5.1 Feature Transformation

* Standarditazion: 'annual premium';
* Rescaling: 'age' and 'vintage' with the MinMaxScaler;
* Encoding: 'gender' and 'region code' with Target Encoding; 'vehicle age' with One Target Encoding; and 'policy sales channel' with Frequency Encoding;

### 5.2 Feature Selection: 

The feature selection was performed using the extra trees classifier algorithm

<img src="image/features_importantes2.png" width="300">

<img src="image/features_importances1.png" width="600">

To work with machine learning algorithms, these columns were then selected:

* 'annual_premium',
* 'vintage',
* 'age',
* 'region_code',
* 'vehicle_damage',
* 'previously_insured',
* 'policy_sales_channel'

[Complete Notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v04_sales_forecast_feature_selection.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)


## 6. Machine Learning Modeling

### 6.1. Comparative Model Performance (with Cross Validation)

<img src="image/cv_result.png" width="300">

<!--

The model chosen for the solution was XGBoost. Despite not having achieved the best result, it ended up being the best option when analyzing the cost/benefit of the solution.

### 6.2. Hyperparameter Fine Tunning

After performing the Fine Tunning process, the model reached a MAPE of 88%.

<img src="img/Hyperparameter Fine Tunning.png" width="500">

The parameters used to achieve these results were:

 * n_estimators: 3000
 * eta: 0.03
 * max_depth: 5
 * subsample: 0.7
 * colsample_bytree: 0.7
 * min_child_weight: 3

[Complete Notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v06_sales_forecast_fine_tunning.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)

## 7. Evaluation

<img src="img/ml_evaluation.png" width= "1000">

[Complete Notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v06_sales_forecast_fine_tunning.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)

## 8. Deployment

* **1. Telegram bot:** the bot receives the Telegram message, validates the information and forwards the data to the Handler API. The code was built using the Flask package and deployed on heroku cloud. [Here](https://github.com/vitorhmf/sales-predict/blob/main/rossmann-telegram-api/rossmann-bot.py) you can check the complete Telegram bot code.
* **2. Handler API:** this API receives the data from the bot, accesses the trained model and returns the prediction to the bot. The code was built using the Flask package and deployed on heroku cloud. [Here](https://github.com/vitorhmf/sales-predict/blob/main/api/handler.py) you can check the complete Handler API code.
* **3. Rossmann Class:** the Rossmann Class runs the developed machine learning model and returns with the requested sales forecast. [Here](https://github.com/vitorhmf/sales-predict/blob/main/api/rossmann/Rossmann.py) you can check the complete class code.

The final solution could be access [here](https://t.me/vitorhmf_rossmann_bot).

<img src="img/bot_telegram.jpg" width="250">


[Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)

## 9. Conclusion

### 9.1. Business Results

The total revenue forecast for the next 6 weeks is presented below, considering the worst and best scenario according to the model. And the detailed sales forecast by store can be consulted through a Telegram bot, available at this [link](https://t.me/vitorhmf_rossmann_bot).

<img src="img/final_result.png" width="300">

### 9.2. Next Steps

* Rerun the CRISP cycle to improve machine learning model results.
* Add new functionality in Telegram bot to improve user experience.

[Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology) 

## 10. References

* [IBM Docs](https://www.ibm.com/docs/en/spss-modeler/18.2.0?topic=dm-crisp-help-overview)
* [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
* [Comunidade DS](https://www.comunidadedatascience.com/)
* [Docs do google script](https://core.telegram.org/bots/api)

[Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)

-->
