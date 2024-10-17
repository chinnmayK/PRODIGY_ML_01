
---

# House Price Prediction Using Linear Regression

## Overview

This project implements a machine learning model to predict house prices based on their square footage, number of bedrooms, and number of bathrooms. The model is trained using the Ames Housing dataset.

## Features

- **Square Footage** (`GrLivArea`)
- **Number of Bedrooms** (`BedroomAbvGr`)
- **Number of Bathrooms** (`FullBath` + `HalfBath`)

## Dataset

- **train.csv**: Contains the training data with house features and the target variable (`SalePrice`).
- **test.csv**: Contains test data for making predictions.

## Requirements

To run this project, you need Python 3.x and the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Download the Dataset**:

   Download the Ames Housing dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place the `train.csv` and `test.csv` files in the project directory.

3. **Run the Model**:

   Execute the model script:

   ```bash
   python model.py
   ```

4. **View Results**:

   The model will output evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²). The predictions for the test set will be saved in `submission.csv`.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

## Visualizations

A scatter plot is generated to compare actual prices vs. predicted prices, helping visualize the model's performance.

---