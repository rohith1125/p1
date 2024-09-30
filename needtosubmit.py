# Import necessary libraries
import pandas as pd
import numpy as np
import random
from pandas.api.types import is_numeric_dtype

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Step 1: Data Loading and Preprocessing
def load_and_preprocess_data(training_file, validation_file):
    """
    Loads and preprocesses the training and validation datasets.

    Parameters:
    - training_file: Path to the training CSV file.
    - validation_file: Path to the validation CSV file.

    Returns:
    - training_data: Preprocessed training DataFrame.
    - validation_data: Preprocessed validation DataFrame.
    """
    # Load the training and validation data
    training_data = pd.read_csv(training_file)
    validation_data = pd.read_csv(validation_file)
    
    # Drop the 'Unnamed: 0' column if it exists
    training_data = training_data.loc[:, ~training_data.columns.str.contains('^Unnamed')]
    validation_data = validation_data.loc[:, ~validation_data.columns.str.contains('^Unnamed')]
    
    # Create the target variable for the training data
    NextDayUp_train = []
    for i in range(len(training_data) - 1):
        if training_data.iloc[i + 1]['Open'] > training_data.iloc[i]['Close']:
            NextDayUp_train.append(1)  # Next day's opening price is greater
        else:
            NextDayUp_train.append(0)  # Not greater
    # For the last day, assign '1' as per project note
    NextDayUp_train.append(1)
    training_data['NextDayUp'] = NextDayUp_train
    
    # Create the target variable for the validation data
    NextDayUp_valid = []
    for i in range(len(validation_data) - 1):
        if validation_data.iloc[i + 1]['Open'] > validation_data.iloc[i]['Close']:
            NextDayUp_valid.append(1)
        else:
            NextDayUp_valid.append(0)
    # For the last day, assign '1'
    NextDayUp_valid.append(1)
    validation_data['NextDayUp'] = NextDayUp_valid
    
    # Remove any missing values
    training_data.dropna(inplace=True)
    validation_data.dropna(inplace=True)
    
    # Convert 'Dividends' and 'Stock Splits' to categorical numeric codes only if they are not purely numeric
    for col in ['Dividends', 'Stock Splits']:
        if col in training_data.columns:
            if not is_numeric_dtype(training_data[col]):
                training_data[col] = training_data[col].astype('category').cat.codes
        if col in validation_data.columns:
            if not is_numeric_dtype(validation_data[col]):
                validation_data[col] = validation_data[col].astype('category').cat.codes
    
    # Data Preprocessing: Normalize continuous features using min-max scaling
    features_to_normalize = [col for col in training_data.columns if is_numeric_dtype(training_data[col]) and col != 'NextDayUp']
    for feature in features_to_normalize:
        min_value = training_data[feature].min()
        max_value = training_data[feature].max()
        # Avoid division by zero
        if max_value - min_value != 0:
            training_data[feature] = (training_data[feature] - min_value) / (max_value - min_value)
            validation_data[feature] = (validation_data[feature] - min_value) / (max_value - min_value)
        else:
            training_data[feature] = 0
            validation_data[feature] = 0
    
    # Remove constant features
    training_data = training_data.loc[:, training_data.nunique() > 1]
    validation_data = validation_data.loc[:, validation_data.columns.isin(training_data.columns)]
    
    # Check if datasets are empty after preprocessing
    if training_data.empty:
        raise ValueError("Training data is empty after preprocessing.")
    if validation_data.empty:
        raise ValueError("Validation data is empty after preprocessing.")
    
    return training_data, validation_data

# Step 2: Implement the Original PRISM Algorithm
def prism_algorithm(data, target_class, min_coverage=5, max_conditions=2):
    """
    Implements the original PRISM algorithm focusing on maximizing rule accuracy.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - target_class: The class label to generate rules for.
    - min_coverage: Minimum number of instances a rule must cover.
    - max_conditions: Maximum number of conditions in a rule.

    Returns:
    - rules: A list of generated rules.
    """
    rules = []
    data_remaining = data.copy()
    
    while len(data_remaining[data_remaining['NextDayUp'] == target_class]) >= min_coverage:
        rule_conditions = []
        data_subset = data_remaining.copy()
        
        while len(rule_conditions) < max_conditions:
            best_condition = None
            best_accuracy = 0
            best_subset = pd.DataFrame()
            features = data_subset.columns.drop(['NextDayUp'])
            
            for feature in features:
                if data_subset[feature].dtype == 'object' or data_subset[feature].nunique() <= 10:
                    # Categorical attribute
                    unique_values = data_subset[feature].unique()
                    for value in unique_values:
                        condition = (data_subset[feature] == value)
                        subset = data_subset[condition]
                        if len(subset) < min_coverage:
                            continue
                        accuracy = np.mean(subset['NextDayUp'] == target_class) * 100
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = (feature, '==', value)
                            best_subset = subset
                else:
                    # Continuous attribute
                    thresholds = data_subset[feature].quantile(np.linspace(0.1, 0.9, 9)).unique()
                    for threshold in thresholds:
                        # Condition: feature <= threshold
                        condition_le = (data_subset[feature] <= threshold)
                        subset_le = data_subset[condition_le]
                        if len(subset_le) < min_coverage or len(subset_le) == len(data_subset):
                            continue
                        accuracy_le = np.mean(subset_le['NextDayUp'] == target_class) * 100
                        if accuracy_le > best_accuracy:
                            best_accuracy = accuracy_le
                            best_condition = (feature, '<=', threshold)
                            best_subset = subset_le
                        
                        # Condition: feature > threshold
                        condition_gt = (data_subset[feature] > threshold)
                        subset_gt = data_subset[condition_gt]
                        if len(subset_gt) < min_coverage or len(subset_gt) == len(data_subset):
                            continue
                        accuracy_gt = np.mean(subset_gt['NextDayUp'] == target_class) * 100
                        if accuracy_gt > best_accuracy:
                            best_accuracy = accuracy_gt
                            best_condition = (feature, '>', threshold)
                            best_subset = subset_gt
            if not best_condition:
                break  # No further improvement
            rule_conditions.append(best_condition)
            data_subset = best_subset
        
        if rule_conditions:
            # Apply all conditions to get the coverage
            condition_mask = pd.Series([True] * len(data_remaining), index=data_remaining.index)
            for feature, operator, value in rule_conditions:
                if operator == '==':
                    condition_mask &= (data_remaining[feature] == value)
                elif operator == '<=':
                    condition_mask &= (data_remaining[feature] <= value)
                elif operator == '>':
                    condition_mask &= (data_remaining[feature] > value)
            coverage = condition_mask.sum()
            if coverage >= min_coverage:
                # Recalculate accuracy based on all conditions
                accuracy = np.mean(data_remaining.loc[condition_mask, 'NextDayUp'] == target_class) * 100
                # Store the rule
                rule = {
                    'conditions': rule_conditions.copy(),
                    'coverage': coverage,
                    'accuracy': accuracy  # Updated accuracy
                }
                rules.append(rule)
                # Remove covered instances from remaining data
                data_remaining = data_remaining[~condition_mask]
            else:
                break  # Coverage below threshold
        else:
            break  # No conditions added to the rule
    return rules

# Step 3: Apply the Rules and Evaluate Performance
def apply_rules(data, rules, target_class=1, default_class=0):
    """
    Applies the generated rules to the dataset to make predictions.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - rules: List of rules to apply.
    - target_class: The class label to predict if rules match.
    - default_class: The class label to predict if no rules match.

    Returns:
    - predictions: A list of predicted class labels.
    """
    predictions = []
    for idx, row in data.iterrows():
        predicted = False
        for rule in rules:
            match = True
            for feature, operator, value in rule['conditions']:
                if operator == '==':
                    if row[feature] != value:
                        match = False
                        break
                elif operator == '<=':
                    if not row[feature] <= value:
                        match = False
                        break
                elif operator == '>':
                    if not row[feature] > value:
                        match = False
                        break
            if match:
                predictions.append(target_class)
                predicted = True
                break
        if not predicted:
            predictions.append(default_class)
    return predictions

def apply_rules_vectorized(data, rules, target_class=1, default_class=0):
    """
    Applies the generated rules to the dataset to make predictions using vectorization.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - rules: List of rules to apply.
    - target_class: The class label to predict if rules match.
    - default_class: The class label to predict if no rules match.

    Returns:
    - predictions: A list of predicted class labels.
    """
    predictions = pd.Series([default_class] * len(data), index=data.index)
    
    for rule in rules:
        condition = pd.Series([True] * len(data), index=data.index)
        for feature, operator, value in rule['conditions']:
            if operator == '==':
                condition &= (data[feature] == value)
            elif operator == '<=':
                condition &= (data[feature] <= value)
            elif operator == '>':
                condition &= (data[feature] > value)
        predictions[condition] = target_class
    return predictions.tolist()

def evaluate_performance(y_true, y_pred):
    """
    Evaluates the performance of the predictions.

    Parameters:
    - y_true: True class labels.
    - y_pred: Predicted class labels.

    Returns:
    - accuracy: Accuracy of predictions.
    - precision: Precision of predictions.
    - recall: Recall of predictions.
    - f1: F1 Score of predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = ((tp + tn) / len(y_true)) * 100 if len(y_true) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\nValidation Performance:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
    return accuracy, precision, recall, f1

# Main Execution Flow
def main():
    training_file = 'Training200.csv'
    validation_file = 'Validation65.csv'
    training_data, validation_data = load_and_preprocess_data(training_file, validation_file)
    
    # Handle class imbalance by oversampling the minority class
    class_counts = training_data['NextDayUp'].value_counts()
    if class_counts.min() / class_counts.max() < 0.5:
        minority_class = class_counts.idxmin()
        samples_to_add = class_counts.max() - class_counts.min()
        minority_samples = training_data[training_data['NextDayUp'] == minority_class]
        oversampled_minority = minority_samples.sample(n=samples_to_add, replace=True, random_state=42)
        training_data = pd.concat([training_data, oversampled_minority], axis=0).reset_index(drop=True)
        training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Oversampled {samples_to_add} instances of class {minority_class}.")
    
    # Define hyperparameter grid
    hyperparams = {
        'min_coverage': [5, 10],
        'max_conditions': [2, 3]
    }
    
    best_accuracy = -1
    best_params = {}
    best_rules = []
    
    print("\nStarting Hyperparameter Tuning...\n")
    for min_cov in hyperparams['min_coverage']:
        for max_cond in hyperparams['max_conditions']:
            print(f"Testing Hyperparameters: min_coverage={min_cov}, max_conditions={max_cond}")
            rules = prism_algorithm(
                data=training_data,
                target_class=1,
                min_coverage=min_cov,
                max_conditions=max_cond
            )
            if not rules:
                print("No rules generated with these hyperparameters.\n")
                continue
            predictions = apply_rules_vectorized(validation_data, rules, target_class=1, default_class=0)
            accuracy, precision, recall, f1 = evaluate_performance(validation_data['NextDayUp'], predictions)
            print(f"Performance: Accuracy={accuracy:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%, F1 Score={f1:.2f}%\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'min_coverage': min_cov,
                    'max_conditions': max_cond
                }
                best_rules = rules
    
    print("Hyperparameter Tuning Completed.\n")
    if best_params:
        print("Best Hyperparameters:")
        print(best_params)
        print(f"Best Accuracy: {best_accuracy:.2f}%\n")
    else:
        print("No valid hyperparameter combinations found.\n")
    
    print("Best Rules:")
    if not best_rules:
        print("No rules generated with the tuned hyperparameters.")
    else:
        for idx, rule in enumerate(best_rules):
            conditions = ' AND '.join([
                f"{feat} {op} {val if isinstance(val, str) else round(val, 4)}"
                for feat, op, val in rule['conditions']
            ])
            print(f"Rule {idx+1}: IF {conditions} THEN NextDayUp = 1 "
                  f"(Coverage: {rule['coverage']}, Accuracy: {rule['accuracy']:.2f}%)")
    
    # Final evaluation with best hyperparameters
    if best_rules:
        final_predictions = apply_rules_vectorized(validation_data, best_rules, target_class=1, default_class=0)
        final_accuracy, final_precision, final_recall, final_f1 = evaluate_performance(validation_data['NextDayUp'], final_predictions)
        print("\nFinal Evaluation Metrics with Best Hyperparameters:")
        print(f"Accuracy: {final_accuracy:.2f}%")
        print(f"Precision: {final_precision:.2f}%")
        print(f"Recall: {final_recall:.2f}%")
        print(f"F1 Score: {final_f1:.2f}%")
    else:
        print("No rules were generated during hyperparameter tuning.")

# Execute the main function
if __name__ == "__main__":
    main()
