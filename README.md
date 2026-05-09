# AI Personal Finance Advisor

A machine learning web app that predicts your monthly expenses, compares them with your spending history, and gives you personalized saving advice all in one place.

---

## What This Project Does

Most people don't realize how much they're overspending until it's too late. This project tries to fix that. You enter a date and a spending category, and the app predicts how much you're likely to spend. It then tells you whether that's too high, okay, or well within budget and gives you specific tips to manage it better.

On top of that, there's a full trend analysis section where you can see your spending patterns across months and categories through charts and a heatmap.

---

## Why I Built This

I wanted to build something that goes beyond just training a model and printing accuracy scores. This project combines machine learning, data analysis, and a real usable interface something you can actually open and interact with. The finance domain made sense because everyone deals with money and budgeting is a genuinely useful problem to solve.

---

## Project Structure

```
project/
│
├── apps.py               # Main Streamlit app
├── utils.py              # Helper function to load data
├── train_model.py        # Script to train and save the ML model
├── data/
│   └── expenses.csv      # Historical expense data
└── saved_model/
    └── model.pkl         # Trained RandomForest model
```

---

## How It Works

Step 1 — The expense data is loaded from a CSV file with three columns: date, category, and amount. Categories are Food, Travel, Shopping, and Bills.

Step 2 — A RandomForestRegressor model is trained on this data. The date is broken into day and month features, and the category is one-hot encoded to make it usable for the model.

Step 3 — When you open the app and select a date and category, the same feature engineering is applied to your input and the model predicts your likely spend.

Step 4 — The predicted amount is compared against your historical average and maximum for that category.

Step 5 — Based on the prediction, the recommendation engine kicks in. Each category has its own high and medium thresholds, and the advice you get is specific to what you're likely to overspend on.

---

## Features

Expense prediction using a trained Random Forest model

Personalized saving advice with category-specific tips

Historical comparison showing your average and max spend

Category-wise bar chart to see where your money goes

Monthly trend line chart to track spending over time

Category x Month heatmap for a detailed breakdown

Raw data table so you can see your full expense history

---

## Tech Stack

Python, Pandas, NumPy for data handling

scikit-learn for the machine learning model

Matplotlib and Seaborn for all the charts

Streamlit for the web interface

Pickle for saving and loading the trained model

---

## How to Run This Locally

Clone the repo and install the dependencies:

```
pip install streamlit scikit-learn pandas matplotlib seaborn
```

First, train the model by running:

```
python train_model.py
```

This will create the saved_model folder and save model.pkl inside it.

Then launch the app:

```
streamlit run apps.py
```

The app will open in your browser automatically.

---

## The Recommendation Logic

The advice engine works on simple but effective thresholds. For example, if your predicted Food spend crosses 400, you get a red alert and a tip to try meal prepping. If it's between 200 and 400, it's a yellow warning. Below 200, you're doing well and the app suggests putting the savings into an emergency fund.

Every category has its own thresholds set based on realistic monthly spending ranges.

---

## What I Learned

Feature engineering on date columns is more useful than it looks. Breaking a date into day and month gave the model meaningful patterns to learn from. Also, building the recommendation engine made me think about how to make ML outputs actually useful to a non-technical person  not just a number, but real advice they can act on.

---
## Output
[AI Personal Finance Advisor.pdf](https://github.com/user-attachments/files/27387435/AI.Personal.Finance.Advisor.pdf)
[AI Personal Finance Advisor1.pdf](https://github.com/user-attachments/files/27387442/AI.Personal.Finance.Advisor1.pdf)

## Author

Riya Singh
github.com/singhriyaaa
linkedin.com/in/riya-singh-58b668244
