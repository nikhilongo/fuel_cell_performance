from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from sklearn.svm import SVR



data = pd.read_csv('Fuel_cell_performance_data-Full.csv')

target_col = 'Target3'
data = data[[target_col] + [col for col in data.columns if col != target_col]]

X = data.drop(columns=[target_col])  
y = data[target_col]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'R2 Score': r2})


svr = SVR()
evaluate_model('Support Vector Machine', svr, X_train, X_test, y_train, y_test)

rf= RandomForestRegressor(random_state=42)
evaluate_model('Random Forest', rf, X_train, X_test, y_train, y_test)

lr = LinearRegression()
evaluate_model('Linear Regression', lr, X_train, X_test, y_train, y_test)

gb = GradientBoostingRegressor(random_state=42)
evaluate_model('Gradient Boosting', gb, X_train, X_test, y_train, y_test)

results_df = pd.DataFrame(results)

print("\nModel Evaluation Results:")
print(results_df)

results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")