import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
class predictor:
    def __init__(self):
        # no need to change the path it will automatically access the dataset in web folder 
        df = pd.read_csv('../Cleaned_Heart_disease.csv')
        df.drop('Has_Prevalent_Stroke', axis=1, inplace=True)
        self.X_train = df.iloc[:, :-1]
        self.y_train = df.iloc[:, -1]
        X_train = self.X_train



    def formdata(self,health_data):
        incoming_data = {
        'Gender': health_data.gender,
        'Age': health_data.age,
        'Is_Smoker': health_data.current_smoker,
        'Cigarettes_Per_Day': health_data.cigs_per_day,
        'Systolic_BP': health_data.sys_bp,
        'Diastolic_BP': health_data.dia_bp,
        'Has_BP_Meds': health_data.bp_meds,
        'Has_Prevalent_Hypertension': health_data.prevalent_hypertension,
        'Glucose': health_data.glucose,
        'Has_Diabetes': health_data.diabetes,
        'BMI': health_data.bmi,
        'Total_Cholestrol': health_data.total_cholestrol
        }

        return incoming_data
    

    def encoding(self, incoming_data):

        self.X_test = pd.DataFrame([incoming_data])

        # Define bin edges and labels for BMI categories
        bin_edges = [0, 18.5, 24.9, 29.9, 100]  # BMI categories: Underweight, Normal weight, Overweight, Obesity
        bin_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

        # Create a new column with BMI categories
        self.X_test['BMI_Category'] = pd.cut(self.X_test['BMI'], bins=bin_edges, labels=bin_labels, right=False)

        self.X_test['Gender'] = self.X_test['Gender'].replace({'male': 1, 'female': 0})
        self.X_test['BMI_Category'] = self.X_test['Gender'].replace({'Underweight': 0, 'Normal weight': 1 , 'Overweight' : 2 , 'Obesity': 3 })
        # self.X_test['Has_Prevalent_Hypertension'] = self.X_test['Has_Prevalent_Hypertension'].replace({'Yes': 1, 'No': 0 })
        # self.X_test['Is_Smoker'] = self.X_test['Is_Smoker'].replace({'Yes': 1, 'No': 0 })
        # self.X_test['Has_BP_Meds'] = self.X_test['Has_BP_Meds'].replace({'Yes': 1, 'No': 0 })

        self.X_test = self.X_test.reindex(columns=['Gender',
            'Age',
            'Is_Smoker',
            'Cigarettes_Per_Day',
            'Systolic_BP',
            'Diastolic_BP',
            'Has_BP_Meds',
            'Has_Prevalent_Hypertension',
            'Glucose',
            'Has_Diabetes',
            'BMI',
            'BMI_Category',                          
            'Heart_Rate',
            'Total_Cholestrol' ])


    
    def prediction(self):
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }

        model = DecisionTreeClassifier(random_state=42)

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

        # Perform grid search to find the best hyperparameters
        grid_search.fit(self.X_train, self.y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions on the testing data using the best model
        y_pred = best_model.predict(self.X_test)
        
        if y_pred[0] == 1 :
            return "High chances of Heart Stroke"
        else:
            return "Low chances of Heart Stroke"



# pred = predictor()
# forminput_data = pred.formdata()
# pred.encoding(forminput_data)
# print(pred.prediction())
