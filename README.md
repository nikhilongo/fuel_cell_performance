# Fuel Cell Performance Model Evaluation

This project involves training and evaluating different machine learning models to predict fuel cell performance using the provided dataset. The models are compared based on their R-squared (R²) scores, which measure the proportion of the variance in the target variable that is predictable from the independent variables.

## Dataset
The dataset used for this project is `Fuel_cell_performance_data-Full.csv`, which contains performance metrics for fuel cells. The target variable is labeled as `Target3`, while the other columns are features.

### Preprocessing
1. The target variable, `Target3`, is moved to the first column for clarity.
2. The features (`X`) are separated from the target variable (`y`).
3. The data is split into training and testing sets using an 80-20 split ratio (70% training, 30% testing).

## Models
The following machine learning models were trained and evaluated:
1. **Support Vector Regression (SVR)**
2. **Random Forest Regressor**
3. **Linear Regression**
4. **Gradient Boosting Regressor**

## Code Workflow
1. **Dataset Loading:**
   The dataset is loaded using Pandas.

2. **Splitting Data:**
   The `train_test_split` function is used to divide the data into training and testing subsets.

3. **Model Evaluation Function:**
   A generic `evaluate_model` function is defined to:
   - Train the model on the training set.
   - Predict the target variable for the test set.
   - Calculate the R² score.
   - Append the results (model name and R² score) to a list for comparison.

4. **Model Training and Evaluation:**
   Each of the specified models is trained and evaluated using the `evaluate_model` function.

5. **Results Comparison:**
   A DataFrame (`results_df`) is created to store the R² scores of all models. The results are printed and saved to a CSV file (`model_results.csv`).

## Requirements
The following Python libraries are required to run the code:
- `pandas`
- `sklearn`

To install these dependencies, use:
```bash
pip install pandas scikit-learn
```

## Usage
1. Place the `Fuel_cell_performance_data-Full.csv` file in the same directory as the script.
2. Run the script using Python:
   ```bash
   python script_name.py
   ```
3. The script will output the R² scores of the models in the console and save the results to `model_results.csv`.

## Output
The script outputs a table of R² scores for each model, helping identify the best-performing model for predicting fuel cell performance.

## Example Output
```text
Model Evaluation Results:
                   Model  R2 Score
0  Support Vector Machine     0.85
1          Random Forest       0.92
2       Linear Regression      0.78
3     Gradient Boosting        0.89

Results saved to 'model_results.csv'
```

## Notes
- Modify the `test_size` parameter in `train_test_split` to adjust the split ratio.
- The `random_state` parameter ensures reproducibility of the results.
- Add or replace models in the `evaluate_model` function as needed for further experimentation.

## License
This project is licensed under the MIT License.

