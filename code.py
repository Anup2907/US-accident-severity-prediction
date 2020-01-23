import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from xgboost              import XGBClassifier
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.ensemble     import GradientBoostingClassifier
import operator

#Reading SAS files
accident = pd.read_sas("accident.sas7bdat")
vehicle = pd.read_sas("vehicle.sas7bdat")
distract = pd.read_sas("distract.sas7bdat")
person = pd.read_sas("person.sas7bdat")

#Number of columns and rows of each variable
accident.shape
vehicle.shape
distract.shape
person.shape

#dataframe info including column count and datatypes
accident.info(verbose=True)
vehicle.info(verbose=True)
distract.info(verbose=True)
person.info(verbose=True)

#Missing value percentage of each variable in every dataset
accident.isnull().sum()/len(accident)*100
vehicle.isnull().sum()/len(vehicle)*100
distract.isnull().sum()/len(distract)*100
person.isnull().sum()/len(person)*100 
#Person dataset has Make, body type, of 3.59% of null values

############################## DATA CONSOLIDATION ##########################################
#common columns in accident and vehicle
a = vehicle.columns.intersection(accident.columns)
print(a)
#common columns in distract and person
b = person.columns.intersection(distract.columns)
print(b)

#Merging the data frames with the CASENUM (ACCIDENT with VEHICLE)
merge_acc_veh = pd.DataFrame()
merge_acc_veh = pd.merge(accident, vehicle, on =['CASENUM'], how='right')
merge_acc_veh.shape

#Merging the data frames with the CASENUM, VEHNO (merge_acc_veh with PERSON)
Injury = pd.DataFrame()
Injury = pd.merge(merge_acc_veh, person, on = ['CASENUM', 'VEH_NO'], how='right')
Injury.shape
Injury.isnull().sum()/len(Injury)*100

############################ DATA FILTERING AND CLEANING ###############################

#Including the rows that has SEAT_POS 'front seat left side'
Injury = Injury.loc[Injury['SEAT_POS'] == 11]

#Including the columns as per business knowledge and intuition and dropping the rest
Injury_final = pd.DataFrame(Injury, columns = ['CASENUM',
'URBANICITY', 
'NUM_INJ',
'ALCOHOL',
'MAN_COLL', 
'TYP_INT',
'WRK_ZONE',
'SPEEDREL',
'REL_ROAD',
'LGT_COND',
'WEATHER',
'INT_HWY',
'MOD_YEAR_x',
'J_KNIFE',
'HAZ_INV',
'TRAV_SP',
'ROLLOVER_y',
'DEFORMED',
'VSURCOND',
'AGE',
'SEX',
'INJ_SEV',
'AIR_BAG',
'EJECTION',
'DRUGS'])

Injury_final.columns
Injury_final.shape

#Including the rows of dependent variable INJ_SEV from 1 to 4
Injury_final = Injury_final.loc[(Injury_final['INJ_SEV'] >= 1) & (Injury_final['INJ_SEV'] <= 4)]

#Fixed Width Binning for Age
bins = [0,17,60,150]
labels = ["teen","adult","elderly"]
Injury_final['Agegroup'] = pd.cut(Injury.AGE, bins=bins, labels=labels)
Injury_final[['AGE','Agegroup']].head(10)

#Calculating vehicle age
Injury_final['Veh_age'] = 2017 - Injury_final['MOD_YEAR_x']

#Dropping AGE, TRAV_SP and MOD_YEAR as we have created new variables
Injury_final = pd.DataFrame(Injury_final.drop(['MOD_YEAR_x','AGE','TRAV_SP'], axis=1))

Injury_final.columns

Injury_final.rename(columns={"CASENUM":"Casenum",
                           "URBANICITY":"Urban",
                           "TYP_INT":"Intersection_type",
                           "REL_ROAD":"Road_loc_of_crash",
                           "INT_HWY":"Interstate_highway",
                           "MAN_COLL":"Manner_of_collision",
                           "LGT_COND":"Light_condition",
                           "WEATHER":"Weather",
                           "ALCOHOL":"Alcohol_involved",
                           "HAZ_INV":"Hazard_involved",
                           "DEFORMED":"Veh_deform",
                           "SPEEDREL":"Driver_speed",
                           "VSURCOND":"Road_surfcond",
                           "NUM_INJ":"Num_of_injured",
                           "AIR_BAG":"Air_bag",
                           "ROLLOVER_y":"Rollover",
                           "WRK_ZONE":"Work_zone",
                           "DRUGS":"Drugs_involved",
                           "SEX":"Sex",
                           "EJECTION":"Ejection",
                           "J_KNIFE":"J_knife",
                           "INJ_SEV":"Injury_sev",
                           "Agegroup":"Age_group"}, inplace=True) 
Injury_final.columns

#Changing column position
Injury_final = Injury_final.reindex(columns=['Casenum', 'Urban', 'Alcohol_involved',
                                             'Manner_of_collision',
       'Intersection_type', 'Work_zone', 'Road_loc_of_crash',
       'Light_condition', 'Weather', 'Interstate_highway','Num_of_injured', 'Driver_speed','J_knife',
       'Hazard_involved', 'Rollover', 'Veh_deform', 'Road_surfcond', 'Sex',
       'Air_bag', 'Ejection', 'Drugs_involved', 'Age_group','Veh_age','Injury_sev'])

#Assigning categorical names to the numbers all variables
Injury_cat = {"Injury_sev": {1:"0", 2:"0", 3:"1", 4:"1"},
              "Sex": {2:"female" , 1:"male", 8:"other", 9:"other"},
              "Veh_age" : {-7981:"0", -7982:"0", -1:"0"},
              "J_knife" : {0:"not_artveh", 1:"no", 2:"yes_firstevent", 3:"yes_subseqevent"},
              "Urban" : {1:"urban", 2:"rural"},
              "Intersection_type" : {1:"no",2:"yes",3:"yes",4:"yes",5:"yes",6:"yes",7:"yes",
                                     10:"yes",98:"other",99:"other"},
              "Work_zone" : {0:"no",1:"yes",2:"yes",3:"yes",4:"yes"},
              "Interstate_highway" : {0:"no",1:"yes",9:"unkown"},
              "Manner_of_collision" : {0:"no",1:"fronttorear",2:"fronttofront",6:"angle",
                                       7:"sideswipe_samedir",8:"sideswipe_oppdir",9:"reartoside",
                                       10:"reartorear",11:"other",98:"other",99:"other"},
              "Light_condition" : {1:"day",2:"dark",3:"dark",6:"dark",4:"dawn",5:"dusk",
                                   7:"other",8:"other",9:"other"},
              "Alcohol_involved" : {1:"yes",2:"no",8:"other",9:"other"},
              "Hazard_involved" : {1:"no",2:"yes"},
              "Veh_deform" : {0:"no",2:"yes",4:"yes",6:"yes",8:"other",9:"other"},
              "Road_surfcond" : {1:"dry",2:"wet",3:"snow",4:"ice",5:"sand",6:"water",7:"oil",
                                 0:"other",8:"other",10:"slush",11:"mud",98:"other",99:"other"},
              "Rollover" : {0:"no",1:"yes",2:"yes",9:"yes"},
              "Air_bag" : {1:"deployed",2:"deployed",3:"deployed",7:"deployed",8:"deployed",
                           9:"deployed",20:"not deployed",28:"switched off",0:"other",
                           97:"other",98:"other",99:"other"},
              "Drugs_involved" : {0:"no",1:"yes",8:"other",9:"other"},
              "Ejection": {0:"no",1:"yes",2:"yes",3:"yes",7:"other",8:"other",9:"other"},
              "Road_loc_of_crash": {1:"onroadway",2:"shoulder",3:"median",4:"onroadside",
                                    5:"outsidetraffic",6:"offroadway",7:"inparking",8:"gore",
                                    10:"separator",11:"cont_leftturn_lane",98:"other",99:"other"},
              "Weather": {0:"nocond",1:"clear",2:"rain",3:"sleet",4:"snow",5:"fog",6:"crosswinds",
                          7:"blowingsand",10:"cloudy",11:"blowingsnow",12:"freezingrain",
                          8:"other",98:"other",99:"other"},
              "Driver_speed" : {0:"no",2:"yes_racing",3:"yes_exceededSL",4:"yes_toofastforcond",
                                5:"yes_noreason",8:"other",9:"other"}}         

#Replacing the dataframe with dictionary "Injury_cat"
Injury_final.replace(Injury_cat, inplace=True)
Injury_final.head()

#Dropping the numerical columns
injury_no_num = pd.DataFrame(Injury_final.drop(['Injury_sev','Casenum','Veh_age','Num_of_injured'], axis=1))

#Converting all the categorical variables to category datatype
injury_no_num = injury_no_num.astype('category')
injury_no_num.dropna(inplace = True)
injury_no_num.dtypes

#Imputing the categorical variable null values with most-frequent
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = SimpleImputer.fit_transform(imputer,injury_no_num.astype(str))

#Label encoding
injury_no_num.dtypes
for col in list(injury_no_num.columns):
    le = preprocessing.LabelEncoder()
    injury_no_num[col] = le.fit_transform(injury_no_num[col].astype(str))
    
injury_no_num.dtypes
injury_no_num = injury_no_num.astype('category')

#dataframe with only the numerical variables
injury_num = pd.DataFrame(Injury_final, columns = ['Injury_sev','Casenum','Veh_age','Num_of_injured'])

#dropping null values of injury_num
injury_num.dropna(inplace = True)

#converting veh_age datatype from category to float
injury_num.dtypes
injury_num["Veh_age"] = injury_num["Veh_age"].astype(float)

#concatenating injury_num and injury_no_num
injury_clean = pd.concat([injury_no_num, injury_num], axis=1)
injury_clean.shape

######################## DESCRIPTIVE STATISTICS ##############################

#Count plots of variables
from matplotlib import pyplot as plt
var = ["Age_group","Injury_sev","Urban","Light_condition","Sex"]

for v in var:
    sns.set(style="darkgrid")
    sns.countplot(x=v, data=injury_clean)
    plt.show()
##Dependent variable Injury_sev is imbalanced, minority class is very low##  

#Relationship between vehicle age and number of injured
ax = sns.scatterplot(x="Veh_age", y="Num_of_injured",data=injury_clean) 

#Boxplots of Vehicle age and Num_of_injured
bx = sns.boxplot(x="Veh_age", data=injury_clean) 
cx = sns.boxplot(x="Num_of_injured", data=injury_clean) 

#Statistics of Veh_age and Num_of_injured
injury_clean.Veh_age.describe()
injury_clean.Num_of_injured.describe()

#One hot encoding using get dummies for numerical machine learning models
Injury_final_drop = injury_clean.drop(columns=['Injury_sev','Casenum','Veh_age','Num_of_injured'])
Injury_dummys = pd.get_dummies(Injury_final_drop, drop_first=True)
Injury_dummys.head()

#concatening dropped columns
injury_no_dummy = injury_clean[['Injury_sev','Casenum','Veh_age','Num_of_injured']]
Injury_finals = pd.concat([injury_no_dummy, Injury_dummys], axis=1)

Injury_finals.dtypes

#Dropping casenum as it is not needed for Tree based models
Injury_nocase = pd.DataFrame(injury_clean.drop(columns=['Casenum']))
Injury_nocase.dropna(inplace=True)

#Splitting the dependent and independent variables
X = pd.DataFrame(Injury_nocase.drop(columns=['Injury_sev']))
Y = pd.DataFrame(Injury_nocase['Injury_sev'], index=None)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, 
                                                    stratify=Y, random_state = 77)

#Converting dataframe to ndarray
X_train = X_train.values
y_train = y_train.values

################### CREATING CLASS FOR MODEL RESULTS ######################

class model_results:
    
    def __init__(self,accuracy,precision,recall,f1,models,auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.models = models
        self.auc = auc
        
    def print_res(self):
        print("_______________________________________________")
        print("\t" + format(self.models))
        print("_______________________________________________")
        print("accuracy:\t {}".format(self.accuracy))
        print("precision:\t {}".format(self.precision))
        print("recall:\t\t {}".format(self.recall))
        print("f1:\t\t {}".format(self.f1))
        print("AUC:\t\t {}".format(self.auc))

#Creating function for best metric       
def optimum_model(results_list, metric):
        accuracies = {}
        precisions = {}
        recalls = {}
        f1s = {}
        aucs = {}
        for result in results_list:
            accuracies[result] = result.accuracy
            precisions[result] = result.precision
            recalls[result] = result.recall
            f1s[result] = result.f1
            aucs[result] = result.auc
        if metric == 'accuracy':
            return max(accuracies.items(), key=operator.itemgetter(1))[0]
        elif metric == 'precision':
            return max(precisions.items(), key=operator.itemgetter(1))[0]
        elif metric == 'recall':
            return max(recalls.items(), key=operator.itemgetter(1))[0]
        elif metric == 'f1':
            return max(f1s.items(), key=operator.itemgetter(1))[0]
        elif metric == 'auc':
            return max(aucs.items(), key=operator.itemgetter(1))[0]                     
            
#Print results function
def print_results(headline, true_value, pred):
    accuracy = accuracy_score(true_value, pred)
    precision = precision_score(true_value, pred, average="binary", pos_label="0")
    recall = recall_score(true_value, pred, average="binary", pos_label="0")
    f1 = f1_score(true_value, pred, average="binary", pos_label="0")
    auc = roc_auc_score(true_value, pred)
    result = model_results(accuracy,precision,recall,f1,auc,headline)
    print("___________________________________________________")
    print("\t" + headline)
    print("___________________________________________________")
    print("accuracy:\t {}".format(accuracy))
    print("precision:\t {}".format(precision))
    print("recall:\t\t {}".format(recall))
    print("f1:\t\t {}".format(f1))
    print("AUC:\t\t {}".format(auc))
    print("___________________________________________________")
    print(classification_report(y_test, samp_prediction))
    print(classification_report_imbalanced(y_test, samp_prediction))
    return result
            
#######################  MODEL BUILDING  ###########################
 
rfc = RandomForestClassifier(random_state = 77)
dtc = DecisionTreeClassifier(random_state = 77)
knr = KNeighborsClassifier(n_neighbors=3)

models = [rfc, dtc, knr]

#Building pipeline model
for mod in models:
    f = type(mod).__name__
    pipeline = make_pipeline(mod)
    model = pipeline.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print()
    print("-------------------------------------------------------------------")
    print(f+" "+'pipeline score {}'.format(pipeline.score(X_test, y_test)))
    print("-------------------------------------------------------------------")
    print(classification_report(y_test, prediction))
    print("-------------------------------------------------------------------")
    print(classification_report_imbalanced(y_test, prediction))
    
#Using SMOTE and Nearmiss with different models
smt = SMOTE(random_state = 4)
nms = NearMiss()
sampler = [smt, nms]
results = []
for classifier in models:
    for samp in sampler:
        s = type(samp).__name__
        c = type(classifier).__name__
        samp_pipeline = make_pipeline_imb(samp, classifier)
        samp_model = samp_pipeline.fit(X_train, y_train)
        samp_prediction = samp_model.predict(X_test)
        p = print_results(s+" "+c, y_test, samp_prediction)
        results.append(p)
 
#printing the optimum model for each metric
metrics = ["accuracy","precision","recall","f1","auc"]
for metric in metrics:
    print("\nMetric:\t" + metric)
    om_acc = optimum_model(results, metric)
    om_acc.print_res()  
    
######################### MODEL EVALUATION ################################  
### As SMOTE() gave good results than Nearmiss(), evaluating all models with 
### 4-fold cross validation ###    

kf = KFold(n_splits=4, random_state=77)
accuracy = []
precision = []
recall = []
f1 = []
auc = []
for mods in models:
    for train, test in kf.split(X_train, y_train):
        b = type(mods).__name__
        pipeline = make_pipeline_imb(SMOTE(random_state = 77), mods)
        model = pipeline.fit(X_train[train], y_train[train])
        prediction = model.predict(X_train[test])
        accuracy.append(pipeline.score(X_train[test], y_train[test]))
        precision.append(precision_score(y_train[test], prediction,average="binary",pos_label="0"))
        recall.append(recall_score(y_train[test], prediction,average="binary",pos_label="0"))
        f1.append(f1_score(y_train[test], prediction,average="binary",pos_label="0"))
        auc.append(roc_auc_score(y_train[test], prediction))
    print("_______________________________________________")
    print(b)
    print("_______________________________________________")
    print("Mean of scores for 4-fold:")
    print("accuracy: {}".format(np.mean(accuracy)))
    print("precision: {}".format(np.mean(precision)))
    print("recall: {}".format(np.mean(recall)))
    print("f1: {}".format(np.mean(f1)))        
    print("AUC: {}".format(np.mean(auc))) 
    
############################### FEATURE IMPORTANCE ################################
    
#converting ndarray back to dataframe
X_train = pd.DataFrame(X_train)

def draw_importance_bar(importances_features):
    fig = plt.figure(figsize=(10,6))
    ax = importances_features.sort_values(ascending=True).plot.barh(color='#046A38', 
                                                                    title='Plot of Important Variables in the Random Forest')
    for rect in ax.patches:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        plt.annotate("{:.3f}".format(x_value), #Display 3 decimal digits of the value
                     (x_value,y_value), # Place label at end of the bar
                     xytext=(5, 0), # Horizontally shift label
                     textcoords="offset points", # Interpret `xytext` as offset in points
                     va='center') # Vertically center label
    interval = round((importances_rf[0] - importances_rf[-1])/10, 2)
    max_limit = round(importances_rf[0] + 2.5*(interval), 2)
    plt.xticks(np.arange(0, max_limit, interval))
    plt.xticks(rotation=90)
    plt.show()
    # Reference : https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
rfc.fit(X_train, y_train)

# Create a pd.Series of features importances and sort it
importances_rf = pd.Series(data=rfc.feature_importances_, 
                           index = X_train.columns)
importances_rf = importances_rf.sort_values(ascending=False)
importances_rf = importances_rf.head(10)

# Make a horizontal bar plot
draw_importance_bar(importances_rf)

### VEHICLE AGE HAS HIGHEST FEATURE IMPORTANCE ACCORDING TO RANDOM FOREST ####################
######################################## THE END #############################################                

