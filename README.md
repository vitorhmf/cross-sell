# Insurance All Company Cross-Sell

### Cross-sell propensity score list built with a Learning to Rank Model

<img src="image/cover_insurance_all.png" width="1000">

## 1. Abstract:

**Disclaimer:** Insurance All is a fictitious company, according to the context presented in this project.

This Data Science project was inspired by this [kaggle challenge] and presents the development of a classification machine learning model, more specifically a learning to rank model, used to generate a propensity score to purchase a new product for a company's customer list. In the commercial arena, this sales strategy is known as Cross-Sell and can be defined as a sales technique that involves selling an additional product or service to an existing customer [Wikipedia](https://en.wikipedia.org /wiki /cross-sell).

In the case of this project, we worked with two sets of data, both composed of Insurance All customers who already have the company's health plan. In the first one, we have the result of a survey carried out with about 380 thousand customers


[ordered list](https://docs.google.com/spreadsheets/d/1vNiaBNN6GXCN-k3ZkEtUqUoJ2NzDeIdRT8PwRNMxO1c/edit?usp=sharing)


<!-- This Data Science project was inspired by a challenge published on [kaggle](https://www.kaggle.com/c/rossmann-store-sales) and presents the construction of a Machine Learning algorithm to predict the 6-week sales of the Rossmann group, which is one of the largest drug store chains in Europe with around 56,200 employees and more than 4000 stores. 

To develop this sales projection, was used a dataset with information from 1115 stores, between 2013-01-01 and 2015-07-31. The trained Regression Algorithm reached 88% of MAPE (Mean Absolute Percentage Error) and the estimated result of the total sales for the period was $287.176.128,00. All the solution was developed with Python language and the complete code is available in this [notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v07_sales_forecast_deploy.ipynb).

The solution was deployed at Heroku Cloud and the sales forecasts can be accessed through a Telegram bot available [here](https://t.me/vitorhmf_rossmann_bot).

<img src="img/bot_telegram.jpg" width="250"> -->

**Keywords:** Python, Regression Model, Random Forest, XGBoost, Scikit Learn, Pandas, Seaborn, Boruta, Flask, Heroku

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

<!-- Insurance All is a company that works with health insurance for its customers and now the product team is analyzing the possibility of offering a new product to its customers: auto insurance.

As with health insurance, customers of this new car insurance plan need to pay an amount annually to Insurance All to obtain an amount insured by the company, intended for the costs of an eventual accident or damage to the vehicle.

Insurance All surveyed 381,109 customers about their interest in joining a new auto insurance product last year. All customers showed interest or not in purchasing auto insurance and these responses were saved in a database along with other customer attributes.

The product team selected 127,000 new customers who did not respond to the survey to participate in a campaign, in which they will be offered the new auto insurance product. The offer will be made by the sales team through phone calls.

However, the sales team has the capacity to make 20,000 calls within the campaign period. -->

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
<!--
* Null values of competitor distance were replaced to 200.000 meters, assuming that there are no competitors.
* Days when the stores were closed, were not considered
* For the missing values in the "Competition Open Since" variable, the approximate year and month were defined as the value from the column Date. 
* The same was done for the variable "Promo 2 Since".

[Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)
-->
 
## 4. Data Understanding

### 4.1. Data Cleaning
<!--
To build an overview of the data, the following steps were performed:
* Change the columns name to sneak_case;
* Shows the data dimensions (rows and columns);
* Check and Fillout NA: for the missing values in the "Competition Open Since" variable, the approximate year and month were defined as the value from the column Date. The same was done for the variable "Promo 2 Since";
* Change types from float64 to int64;


### 4.2 Data Descriptive: 

A quick descriptive analysis of numerical and categorical variables was performed.
 
**Numerical Attributes:**

<img src="img/num_attributes.png" width="800">

**Categorical Attributes:**

<img src="img/cat_attributes.png" width="800">
 
### 4.3. Feature Engineering

Before performing the feature engineering, a mental map was created to evaluate the relationship between the sales phenomenon and the agents that act on it, as well as the attributes of each agent.

<img src="img/MindMapHypothesis.png" width="1000">

From this mental map, business hypotheses were created in order to develop the understanding of the case and raise new variables that are important to derive from the original dataset for the creation of the machine learning model.
 
In this step, the following features were created:
* Features derived from the Date variable: Year, Month, Day, Week of Year, Year Week. 
* Assortment: a = 'basic'; b = 'extra'; c = 'extended'
* State Holiday: a = 'public holiday; b = 'easter holiday'; c = 'christmas
* Other Features: “Competition Since” and “Promo Since”

### 4.4. Data Filtering

* Filtered the rows for open stores.
* Filtered the rows for sales greater than zero.
* Exclude columns already used to create new features.
* Exclude columns with a single value.

### 4.5. Exploratory Data Analysis

In the data exploration, univariate, bivariate and multivariate analyzes were performed. Of the business insights obtained in this phase, two stood out for presenting different results than expected:

**Stores with closer competitors sell more:**

<img src="img/competition_distance.png" width="800">

**Stores with longer promotions sell less:**

<img src="img/promo_time_week.png" width="800">

[Complete Notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v02_sales_forecast_eda.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)
 
## 5. Data Preparation

* Standarditazion: not used because none of the variables showed a normal curve;
* Rescaling: in numerical variables, the MinMax Scaler and Robust Scaler methods were used to balance the range of each variable
* Encoding - applied to categorical variables
* Nature Transformation - for cyclic variables such as month, day and week a sine and cosine transformation was applied
* Feature Selection: the variables to be used in the machine learning model were selected using the Boruta algorithm

[Complete Notebook](https://github.com/vitorhmf/sales-predict/blob/main/notebooks/v04_sales_forecast_feature_selection.ipynb) | [Back to the top](https://github.com/vitorhmf/sales-predict#2-methodology)

## 6. Machine Learning Modeling

### 6.1. Comparative Model Performance (with Cross Validation)

<img src="img/Real Performance - Cross Validation.png" width="500">

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
