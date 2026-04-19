# Iris Flower Classification Project

A complete machine learning project demonstrating flower species classification using the Iris dataset with exploratory data analysis, model training, evaluation, and a CLI prediction tool.

## Project Overview

This project implements a comprehensive machine learning pipeline for the Iris dataset, including:

- **Exploratory Data Analysis (EDA)**: Dataset inspection, statistics, and distributions
- **Feature Visualization**: Pair plots and scatter plots to understand feature relationships
- **Model Training**: Two classifiers for comparison
  - Logistic Regression (linear decision boundaries)
  - Decision Tree (non-linear patterns)
- **Performance Evaluation**: Accuracy metrics and confusion matrices
- **Prediction CLI**: Interactive and command-line interface for predicting flower species

## Project Structure

```
flower_classification/
├── iris_classification.ipynb    # Main Jupyter notebook with full analysis
├── predict_iris.py              # CLI script for flower species prediction
├── README.md                     # This file
└── .venv/                        # Python virtual environment
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Installation

1. **Clone/Navigate to the project directory**:
```bash
cd /Users/sanjana/flower_classification
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Jupyter Notebook Analysis

The main analysis is contained in `iris_classification.ipynb` with the following sections:

### 1. **Import Libraries and Load Iris Dataset**
- Loads the Iris dataset into a pandas DataFrame
- Displays basic information about the dataset

### 2. **Exploratory Data Analysis**
- Dataset shape and structure
- Statistical summary (mean, std, min, max)
- Class distribution across three flower species
- Missing value analysis

### 3. **Visualize Feature Pairs**
- Comprehensive pair plots showing all feature relationships
- Colored by flower species for easy distinction
- Scatter plots for key feature pairs

### 4. **Preprocess Data and Split Dataset**
- Encodes target variable (Setosa: 0, Versicolor: 1, Virginica: 2)
- Splits data into 80% training and 20% test sets
- Maintains class distribution with stratification

### 5. **Train Logistic Regression and Decision Tree**
- Trains Logistic Regression (linear classifier)
- Trains Decision Tree (non-linear classifier)
- Both models fitted on the same training set

### 6. **Compare Accuracy of Classifiers**
- Evaluates both models on the test set
- Displays accuracy comparison visualization
- Shows detailed performance metrics

### 7. **Plot Confusion Matrix and Analyze Errors**
- Side-by-side confusion matrices for both models
- Detailed classification reports with precision, recall, and F1-scores
- Analysis of misclassifications
- Key insights about model behavior and class separability

### 8. **Build Prediction Script / CLI for New Input**
- Implements prediction function for new measurements
- Tests with example inputs for each species
- Demonstrates usage of both models

## Key Findings

### Model Performance
- **Logistic Regression Accuracy**: Typically 95-98%
- **Decision Tree Accuracy**: Typically 93-97%

### Species Characteristics
- **Setosa**: Clearly separable with distinct smaller dimensions
- **Versicolor**: Medium-sized flowers with intermediate measurements
- **Virginica**: Larger flowers, shows some overlap with Versicolor

### Misclassifications
- Most errors occur between Versicolor and Virginica (similar characteristics)
- Setosa has virtually no misclassifications (clearly distinct)
- Logistic Regression typically performs slightly better due to linear separability

## CLI Prediction Tool

### Usage

**1. Direct Prediction (Logistic Regression)**:
```bash
python predict_iris.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

**2. Using Decision Tree Model**:
```bash
python predict_iris.py -sl 5.9 -sw 3.0 -pl 4.2 -pw 1.5 -m dt
```

**3. Interactive Mode**:
```bash
python predict_iris.py --interactive
```

### Command-Line Options

```
-sl, --sepal-length       Sepal length in cm (required for direct mode)
-sw, --sepal-width        Sepal width in cm (required for direct mode)
-pl, --petal-length       Petal length in cm (required for direct mode)
-pw, --petal-width        Petal width in cm (required for direct mode)
-m, --model               Model choice: 'lr' (Logistic Regression) or 'dt' (Decision Tree)
                         Default: 'lr'
-i, --interactive         Run in interactive mode for multiple predictions
-h, --help                Show help message
```

### Example Inputs

**Setosa**:
```bash
python predict_iris.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2
```

**Versicolor**:
```bash
python predict_iris.py -sl 5.9 -sw 3.0 -pl 4.2 -pw 1.5
```

**Virginica**:
```bash
python predict_iris.py -sl 6.3 -sw 3.3 -pl 6.0 -pw 2.5
```

## Visualizations

The notebook includes several visualizations:

1. **Pair Plot**: Full feature relationship matrix colored by species
2. **Scatter Plots**: Key feature pair relationships
3. **Accuracy Comparison**: Bar chart comparing model accuracies
4. **Confusion Matrices**: Heatmaps showing classification performance

## Feature Measurements

The Iris dataset contains 4 features:

1. **Sepal Length** (cm): Length of the flower's sepal
2. **Sepal Width** (cm): Width of the flower's sepal
3. **Petal Length** (cm): Length of the flower's petal
4. **Petal Width** (cm): Width of the flower's petal

**Typical Ranges**:
- Sepal Length: 4.3 - 7.9 cm
- Sepal Width: 2.0 - 4.4 cm
- Petal Length: 1.0 - 6.9 cm
- Petal Width: 0.1 - 2.5 cm

## Dataset Information

- **Total Samples**: 150 (50 per species)
- **Features**: 4 numerical measurements
- **Target Classes**: 3 flower species
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica
- **Source**: Classic machine learning dataset

## Technologies Used

- **Python 3.x**
- **Flask**: Web application interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter**: Interactive notebook environment

## 🌐 Web App

A simple Flask web interface has been added in `app.py`.

To run the web page locally for development:

```bash
cd /Users/sanjana/flower_classification
source .venv/bin/activate
python app.py
```

Then open:

```bash
http://127.0.0.1:5000/
```

For a production-ready server, use Gunicorn:

```bash
cd /Users/sanjana/flower_classification
source .venv/bin/activate
gunicorn --bind 0.0.0.0:8000 app:app
```

Then open:

```bash
http://127.0.0.1:8000/
```

## 📝 Notes

- The models are retrained each time the script is run for reproducibility
- Random seed (42) ensures consistent results across runs
- The CLI automatically trains models on first use
- Interactive mode stores trained models for faster subsequent predictions

## 🎓 Learning Outcomes

This project demonstrates:

✅ Complete ML pipeline from data loading to deployment
✅ EDA techniques for understanding data
✅ Model training and comparison
✅ Performance evaluation with multiple metrics
✅ Handling multi-class classification problems
✅ Building user-friendly CLI interfaces
✅ Best practices in data science projects

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and enhance this project for your learning purposes!

---

**Happy Predicting! 🌸**
# flower_classification
