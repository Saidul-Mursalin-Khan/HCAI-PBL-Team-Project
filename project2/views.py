from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from palmerpenguins import load_penguins

def index(request):
    # Load Palmer Penguins dataset
    df = load_penguins()
    df = df.dropna()  # remove missing values
    
    # Select features and target
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    y = df['species']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define possible max leaves and lambda
    max_leaves_options = [3, 5, 7, 10, 14, 20]
    lambda_value = 0.1

    best_score = -1
    best_tree = None
    best_leaves = None
    chosen_lambda = lambda_value

    for leaves in max_leaves_options:
        tree = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=42)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        acctest = accuracy_score(y_test, y_pred)
        
        score = acctest + lambda_value * leaves
        if score > best_score:
            best_score = score
            best_tree = tree
            best_leaves = leaves

    context = {
        'title': 'Project 2: Decision Tree with Regularization',
        'accuracy': accuracy_score(y_test, best_tree.predict(X_test)),
        'n_leaves': best_leaves,
        'lambda_value': chosen_lambda,
    }
    
    # Train a decision tree
    # tree = DecisionTreeClassifier(random_state=42)
    # tree.fit(X_train, y_train)
    
    # Predict on test set and calculate accuracy
    # y_pred = tree.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    
    # Number of leaves in the tree
    # n_leaves = tree.get_n_leaves()
    
    # context = {
    #     'title': 'Project 2: Decision Tree for Palmer Penguins',
    #     'accuracy': accuracy,
    #     'n_leaves': n_leaves,
    # }
    
    return render(request, 'project2/index.html', context)