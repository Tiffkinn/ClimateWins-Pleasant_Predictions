#allows for easy testing of different numbers of hidden layers, nodes, iterations, and tolerances and returns: results, y_train_pred, y_test_pred
# To call: results, y_train_pred, y_test_pred = train_and_evaluate_mlp(X_train, y_train, X_test, y_test)


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, results=None):
    # Create the results DataFrame if it doesn't exist
    if results is None:
        results = pd.DataFrame(columns=[
            'Hidden Layers', 'Layer Sizes', 'Max Iterations', 
            'Tolerance', 'Training Accuracy', 'Test Accuracy'
        ])

    try:
        # User inputs for MLP parameters
        num_hidden_layers = int(input("Enter the number of hidden layers: "))
        layer_sizes = tuple(int(x) for x in input("Enter the number of nodes in each layer (comma-separated): ").split(','))
        max_iter = int(input("Enter the number of maximum iterations: "))
        tolerance = float(input("Enter the tolerance: "))
        
        # Initialize and fit the MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=max_iter, tol=tolerance)
        clf.fit(X_train, y_train)
    
        # Calculate training accuracy
        y_train_pred = clf.predict(X_train)
        training_accuracy = accuracy_score(y_train, y_train_pred)
        print("Training Accuracy:", training_accuracy)
    
        # Calculate test accuracy
        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("Test Accuracy:", test_accuracy)

        # Store both accuracies in the results DataFrame
        new_row = {
            'Hidden Layers': num_hidden_layers,
            'Layer Sizes': layer_sizes,
            'Max Iterations': max_iter,
            'Tolerance': tolerance,
            'Training Accuracy': training_accuracy,
            'Test Accuracy': test_accuracy
        }
        new_row_df = pd.DataFrame([new_row])
        results = pd.concat([results, new_row_df], ignore_index=True)

    except ValueError as e:
        print(f"Error in input: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return results, None, None

    return results, y_train_pred, y_test_pred
