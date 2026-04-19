#!/usr/bin/env python3
"""
Iris Flower Species Prediction CLI

This script provides a command-line interface to predict Iris flower species
based on measurements. It supports both Logistic Regression and Decision Tree models.

Usage:
    python predict_iris.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
    python predict_iris.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2 --model dt
    python predict_iris.py --interactive
"""

import argparse
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_models():
    """Train and return both Logistic Regression and Decision Tree models."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, 
                                               random_state=42, stratify=y)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    
    return lr_model, dt_model


def predict_species(sepal_length, sepal_width, petal_length, petal_width, model='lr'):
    """
    Predict Iris species for given measurements.
    
    Parameters:
    -----------
    sepal_length : float
        Length of the sepal in cm
    sepal_width : float
        Width of the sepal in cm
    petal_length : float
        Length of the petal in cm
    petal_width : float
        Width of the petal in cm
    model : str
        Model to use ('lr' for Logistic Regression, 'dt' for Decision Tree)
    
    Returns:
    --------
    tuple : (species_name, model_name)
    """
    
    # Create input array
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Use trained models (train on first call)
    if not hasattr(predict_species, 'lr_model'):
        predict_species.lr_model, predict_species.dt_model = train_models()
    
    # Select model
    selected_model = predict_species.lr_model if model == 'lr' else predict_species.dt_model
    model_name = 'Logistic Regression' if model == 'lr' else 'Decision Tree'
    
    # Make prediction
    prediction = selected_model.predict(input_features)[0]
    
    # Map prediction to species name
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    return species_map[prediction], model_name


def validate_input(sepal_length, sepal_width, petal_length, petal_width):
    """Validate that input measurements are positive numbers."""
    try:
        sl = float(sepal_length)
        sw = float(sepal_width)
        pl = float(petal_length)
        pw = float(petal_width)
        
        if any(x <= 0 for x in [sl, sw, pl, pw]):
            raise ValueError("All measurements must be positive numbers")
        
        return sl, sw, pl, pw
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")


def interactive_mode():
    """Run the predictor in interactive mode."""
    print("\n" + "="*70)
    print("IRIS FLOWER SPECIES PREDICTOR - Interactive Mode")
    print("="*70)
    print("\nEnter flower measurements in centimeters.")
    print("Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            print("-" * 70)
            sepal_length = input("Sepal length (cm): ").strip()
            
            if sepal_length.lower() in ['quit', 'exit']:
                print("\nThank you for using the Iris Predictor!")
                break
            
            sepal_width = input("Sepal width (cm): ").strip()
            petal_length = input("Petal length (cm): ").strip()
            petal_width = input("Petal width (cm): ").strip()
            
            # Validate inputs
            sl, sw, pl, pw = validate_input(sepal_length, sepal_width, 
                                           petal_length, petal_width)
            
            # Make predictions with both models
            print("\nPredictions:")
            print("-" * 70)
            
            for model in ['lr', 'dt']:
                species, model_name = predict_species(sl, sw, pl, pw, model=model)
                print(f"  {model_name:25s}: {species}")
            
            print("-" * 70 + "\n")
            
        except ValueError as e:
            print(f"Error: {e}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Iris Flower Species Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_iris.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
  python predict_iris.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2 --model dt
  python predict_iris.py --interactive
        """
    )
    
    parser.add_argument('--sepal-length', '-sl', type=float,
                       help='Sepal length in cm')
    parser.add_argument('--sepal-width', '-sw', type=float,
                       help='Sepal width in cm')
    parser.add_argument('--petal-length', '-pl', type=float,
                       help='Petal length in cm')
    parser.add_argument('--petal-width', '-pw', type=float,
                       help='Petal width in cm')
    parser.add_argument('--model', '-m', choices=['lr', 'dt'], default='lr',
                       help='Model to use: lr (Logistic Regression) or dt (Decision Tree). Default: lr')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Run interactive mode if requested
    if args.interactive:
        interactive_mode()
        return
    
    # Check if all required arguments are provided
    if not all([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]):
        parser.print_help()
        sys.exit(1)
    
    try:
        # Validate inputs
        sl, sw, pl, pw = validate_input(args.sepal_length, args.sepal_width,
                                       args.petal_length, args.petal_width)
        
        # Make prediction
        species, model_name = predict_species(sl, sw, pl, pw, model=args.model)
        
        # Print results
        print("\n" + "="*70)
        print("IRIS FLOWER SPECIES PREDICTION")
        print("="*70)
        print(f"\nInput Measurements:")
        print(f"  Sepal Length: {sl} cm")
        print(f"  Sepal Width:  {sw} cm")
        print(f"  Petal Length: {pl} cm")
        print(f"  Petal Width:  {pw} cm")
        print(f"\nModel: {model_name}")
        print(f"Predicted Species: {species}")
        print("="*70 + "\n")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
