# Predicting the road accident severity depending on weather, vehicle, and road conditions with ML models using CRISP-DM methodology
<img src = https://user-images.githubusercontent.com/56169217/74563646-dd6ba580-4f32-11ea-8c6b-b6ff475eb661.PNG width="400" height="400" />

**US-accident-severity**

One of the primary objectives of the National Highway Traffic Safety Administration (NHTSA) is to reduce the human toll and property damage that motor vehicle traffic crashes inflict on our society. Crashes each year result in thousands of lives lost, hundreds of thousands of injured victims, and billions of dollars in property damage. Accurate data are required to support the development, implementation, and assessment of highway safety programs aimed at reducing this toll. NHTSA uses data from many sources, including the Crash Report Sampling System (CRSS). CRSS is a sample of police-reported crashes involving all types of motor vehicles, pedestrians, and cyclists, ranging from property-damage-only crashes to those that result in fatalities. CRSS is used to estimate the overall crash picture, identify highway safety problem areas, measure trends, drive consumer information initiatives, and form the basis for cost and benefit analyses of highway safety initiatives and regulations. The CRSS obtains its data from a nationally representative probability sample selected from the more than seven million police-reported crashes which occur annually. Although various sources suggest that there are many more crashes that are not reported to the police, the majority of these unreported crashes involve only minor property damage and no significant personal injury. By restricting attention to police-reported crashes, the CRSS concentrates on those crashes of greatest concern to the highway safety community and the general public. 
![car-accident-white-background-flat-style-vector-illustration-95784387](https://user-images.githubusercontent.com/56169217/74563674-e9effe00-4f32-11ea-8fbb-1906d8e90141.jpg)

**Determining Business Objectives**
To predict and identify crucial factors that contribute Person Injury severity.
To investigate the factors that can minimize accidents.
To identify and recommend issues that can improve safety and overall traffic situation.

**Assessing the Situation**
Around 40,000 people lost their lives to car crashes in 2018.
About 4.5 million people were seriously injured in crashes last year.

**Data Mining Goals**
To prepare data to achieve best accurate result.
To build suitable models to predict the severity of fatalities.
To recommend precautionary initiative based on our results. 

**Business and Data Understanding**

Accident file: Alcohol Involved in Crash, Atmospheric Conditions, Crash time (Hour), First Harmful Event, Light Condition, Manner of Collision, Number of Injured in Crash. 

Accident: (54969, 51) rows and columns

Vehicle file: Areas of Impact- Initial Contact Point, Body Type, Driver Drinking in Vehicle, Hit and Run, Number of Injured in Vehicle, Maximum Injury Severity in Vehicle, Most Harmful Event, Vehicle Model Year. 

Vehicle: (97625, 87) rows and columns

Distract file: Person demographic details. 

Distract: (97657, 11) rows and columns

Person file: Age, Alcohol Test Status, Ejection, Injury Severity, Sex. 

Person: (138913, 61) rows and columns. 

Joining Accident with Vehicle using primary key "Accident case number" - (Right Join gives all the vehicle information involved in the accidents)

Joining the merged Accident and Vehicle with Person using "Accident case number and Vehicle number" - (Right Join gives all the Person details who are involved in the vehicle accidents)

The resultant "Injury" file can be used to explain the business question of predicting injury severity.

**Data Preparation**

After reading the meta data description and brainstorming about every feature from 196 features in CRSS manual and understanding the business involved, we have considered 24 features that are important in achieving the business objective of predicting injury severity of person. The features are:

![Capture](https://user-images.githubusercontent.com/56169217/74565896-d98e5200-4f37-11ea-9821-d3cbf7c2f91a.PNG)

Vehicle age and Age group are newly created features. Vehicle age is obtained by subtracting current year minus vehicle model year. Age group is obtained by fixed width numerical binning.

Considered the dependent variable "Injury_sev" as binary. classified as high severity and low severity. Filtered the data that has the driver seat position front left side because US has most vehicles with driver seat left side. That gave us finally with 26367 rows and 24 columns.

We have classified the unknown/notrecording/NA categories as "other" category that has bad data. Also some of the features has many categories which we will end up with many features during one hot encoding, hence we have categorized as classes combining some of the sub categories. Example: Light_Condition has dark_lowlight, dark_heavylight, dark_nolight as category "dark".

There are very less percentage of rows (1.64%) with missing data which we have dropped as imputing would change the variation of features. 

**Data Exploration**

The dependent variable is highly imbalanced. The minority class is very low. We need to balance them by sampling techniques, Please refer below plot:

![injury_sev](https://user-images.githubusercontent.com/56169217/74578475-7eba2200-4f5a-11ea-9ceb-30b2ef5c98ca.PNG)

Checking if there is any relationship between the vehicle age and the nuber of persons getting injured in an accident:

![veh_age_num_of_inj](https://user-images.githubusercontent.com/56169217/74579009-77484800-4f5d-11ea-81d5-48e05c98a330.PNG)

There is no clear non-linear relation between vehicle age and number of injured persons

Checking if there are any outliers in the features Vehicle_age and Num_of_injured:

![box_plot](https://user-images.githubusercontent.com/56169217/74579645-b8daf200-4f61-11ea-8482-86b3ec922568.PNG)

There are few vehicles of age more than 25 years old. As per the US law, any vehicle which is more than 25 years old should go through additional testing and federalization process. But here in the box plots we cannot remove the outliers as old vehicles might be the reason for high injury severity during accidents. Also sometimes due to mass collision of vehicles, especially due to extreme weather conditions, number of injured persons could be high. Hence we are not removing any further data.

**Modeling**

As the problem is classification problem, I came up with the following three models:

1. Random Forest Classifier
2. Decision Tree Classifier
3. K-nearest Neighbours 

The results are:

![model](https://user-images.githubusercontent.com/56169217/74582388-a328f500-4f80-11ea-824d-27b796feeeca.PNG)

As it is evident that the dependent variable "injury_sev" is not balanced, SMOTE technique which upsamples the minority class and Nearmiss technique which downsamples the majority class, and these techniques are used with three models to see accuracy, precision and recall values. Hence built a pipeline for SMOTE/Nearmiss and models. 

1. RandomForest using SMOTE/Nearmiss
2. Decision Tree using SMOTE/Nearmiss
3. K-Nearest Neighbours using SMOTE/Nearmiss

Let's look at the below table to see how these both models performed:

![smote_near](https://user-images.githubusercontent.com/56169217/74593926-3ea87d00-4ff6-11ea-82f0-5e11184ebf6f.PNG)

From the above results, SMOTE gave better results overall comparing with the Nearmiss.

**Model Evaluation**

Now, let's evaluate with K- fold cross validation, I took 4 folds and built a pipeline to run each model with SMOTE. Let's see how we got the results after cross validation 

![k-fold](https://user-images.githubusercontent.com/56169217/74594079-dbb7e580-4ff7-11ea-944f-0b28b6448649.PNG)

Overall SMOTE Random Forest Classifier stands out to be the best model in terms of accuracy, precision and recall.

Let's look at the feature importances from RandomForest Model to know what are the important features that are contributing for injury severity:

![feature](https://user-images.githubusercontent.com/56169217/74595281-46bbe900-5005-11ea-9190-75894fe79903.PNG)

**Conclusion**

From the feature importance plot, we can conclude that Vehicle age, Number of persons injured, Weather conditions, Manner_of_collision, Light conditions are the top 5 features that are contributing to cause injury severity. It's important to check the vehicle conditions thoroughly and be careful while driving in adverse weather conditions.

Please refer my Python notebook for more details - https://github.com/Anup2907/US-accident-severity/blob/master/US_Accident_Severity.ipynb











