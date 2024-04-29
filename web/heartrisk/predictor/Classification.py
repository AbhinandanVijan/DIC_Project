import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
class predictor:
    def __init__(self):
        df = pd.read_csv('/Users/krishnanand/Downloads/web/Cleaned_Heart_disease.csv')
        df.drop('Has_Prevalent_Stroke', axis=1, inplace=True)
        self.X_train = df.iloc[:, :-1]
        self.y_train = df.iloc[:, -1]
        X_train = self.X_train



    def formdata(self):
        # Hardcoded sample data for prediction
        incoming_data = {
            'Gender': 'Female', 
            'Age': 55,  
            'Is_Smoker': 'Yes', 
            'Cigarettes_Per_Day': 5,  
            'Systolic_BP': 1,  
            'Diastolic_BP': 85,  
            'Has_BP_Meds': 'Yes',  
            'Has_Prevalent_Hypertension': 'Yes', 
            'Glucose': 98, 
            'Has_Diabetes': 0, 
            'BMI': 26.5, 
            'Heart_Rate': 77, 
            'Total_Cholestrol': 230, 
        }

        return incoming_data
    

    def encoding(self, incoming_data):

        self.X_test = pd.DataFrame([incoming_data])

        # Define bin edges and labels for BMI categories
        bin_edges = [0, 18.5, 24.9, 29.9, 100]  # BMI categories: Underweight, Normal weight, Overweight, Obesity
        bin_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

        # Create a new column with BMI categories
        self.X_test['BMI_Category'] = pd.cut(self.X_test['BMI'], bins=bin_edges, labels=bin_labels, right=False)

        self.X_test['Gender'] = self.X_test['Gender'].replace({'Male': 1, 'Female': 0})
        self.X_test['BMI_Category'] = self.X_test['Gender'].replace({'Underweight': 0, 'Normal weight': 1 , 'Overweight' : 2 , 'Obesity': 3 })
        self.X_test['Has_Prevalent_Hypertension'] = self.X_test['Has_Prevalent_Hypertension'].replace({'Yes': 1, 'No': 0 })
        self.X_test['Is_Smoker'] = self.X_test['Is_Smoker'].replace({'Yes': 1, 'No': 0 })
        self.X_test['Has_BP_Meds'] = self.X_test['Has_BP_Meds'].replace({'Yes': 1, 'No': 0 })

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
            return 1
        else:
            return 0



pred = predictor()
forminput_data = pred.formdata()
pred.encoding(forminput_data)
print(pred.prediction())
