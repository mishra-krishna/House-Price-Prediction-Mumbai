# %%
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np






# %%
df = pd.read_csv(r'C:\Users\Krishna\Documents\codes\House Price Prediction Mumbai\data\mumbai_cleaned.csv')

# %%
df.head()

# %%
target = df['price']

features = df.drop('price', axis=1)


# %%


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=80)

# Print the shapes of the training and test sets
# print("Training set shape:", X_train.shape)
# print("Test set shape:", X_test.shape)


# %%

def linear_regression(X_train, X_test, y_train, y_test, X_pred):
    # Create a linear regression model
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    y_pred_user = model.predict(X_pred)

    
    mse = r2_score(y_test, y_pred)
    
    
    
    return mse , y_pred_user


# %%


def decision_tree_regressor(X_train, X_test, y_train, y_test, X_pred):
    
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4]  
    }

    dt_regressor = DecisionTreeRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_regressor = grid_search.best_estimator_

    # Predict the target variable for the test data
    y_pred = best_regressor.predict(X_test)
    
    # Predict the target variable for the user-provided data
    y_pred_user = best_regressor.predict(X_pred)

    # Calculate Mean Squared Error for test data
    mse = r2_score(y_test, y_pred)

    return mse, y_pred_user


# %%

def random_forest_regressor(X_train, X_test, y_train, y_test, X_pred):
        # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],       # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],   # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4]      # Minimum samples required at each leaf node
    }
    
    # Create a random forest regressor model
    model = RandomForestRegressor()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='r2', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    
    # Fit the best model on the training data
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    # Predict the target variable for the user-provided data
    y_pred_user = best_model.predict(X_pred)
    
    # Evaluate the model using R-squared
    mse = r2_score(y_test, y_pred)

    
    return mse, y_pred_user


# %%

def support_vector_regressor(X_train, X_test, y_train, y_test, X_pred):
    # Define the hyperparameters grid
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf']
    }

    # Create a support vector regressor model
    model = SVR()

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model on the training data with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_regressor = grid_search.best_estimator_

    # Predict the target variable for the test data
    y_pred = best_regressor.predict(X_test)

    # Predict the target variable for the user-provided data
    y_pred_user = best_regressor.predict(X_pred)

    mse = r2_score(y_test, y_pred)

    return mse, y_pred_user

a,b = decision_tree_regressor(X_train, X_test, y_train, y_test, np.array([1000, 2, 2, 1]).reshape(1, -1))
print(a,b)


# %%
