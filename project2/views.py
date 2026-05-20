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
    
    # Train a decision tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    
    # Predict on test set and calculate accuracy
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Number of leaves in the tree
    n_leaves = tree.get_n_leaves()
    
    context = {
        'title': 'Project 2: Decision Tree for Palmer Penguins',
        'accuracy': accuracy,
        'n_leaves': n_leaves,
    }
    
    return render(request, 'project2/index.html', context)