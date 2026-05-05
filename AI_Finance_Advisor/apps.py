import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data


# LOAD MODEL + DATA

model = pickle.load(open("saved_model/model.pkl", "rb"))  # read binary

# Load full dataset for analysis
df = load_data("data/expenses.csv")


# PERSONALIZED RECOMMENDATION ENGINE
# Returns dynamic advice based on predicted amount + category


def get_recommendation(category, predicted_amount):
    recommendations = []

    # Category-specific thresholds
    thresholds = {
        "Food":     {"high": 400,  "medium": 200},
        "Travel":   {"high": 1000, "medium": 500},
        "Shopping": {"high": 800,  "medium": 400},
        "Bills":    {"high": 1500, "medium": 800},
    }

    limit = thresholds.get(category, {"high": 800, "medium": 400})

    if predicted_amount >= limit["high"]:
        recommendations.append(
            f"🔴 HIGH ALERT: Predicted {category} spend of ₹{predicted_amount:.0f} is above your safe limit.")
        if category == "Food":
            recommendations.append(
                "💡 Try meal prepping at home and reduce dining out this month.")
        elif category == "Travel":
            recommendations.append(
                "💡 Consider local travel or postpone non-essential trips.")
        elif category == "Shopping":
            recommendations.append(
                "💡 Apply a 48-hour rule — wait 2 days before any purchase above ₹500.")
        elif category == "Bills":
            recommendations.append(
                "💡 Review your subscriptions. Cancel unused services to cut bill costs.")

    elif predicted_amount >= limit["medium"]:
        recommendations.append(
            f"🟡 MODERATE: Predicted {category} spend of ₹{predicted_amount:.0f} is within range but watch out.")
        recommendations.append(
            "💡 You're on track — just avoid impulse spending for the rest of the month.")

    else:
        recommendations.append(
            f"🟢 GOOD: Predicted {category} spend of ₹{predicted_amount:.0f} is low. Keep it up!")
        recommendations.append(
            "💡 Consider moving the saved amount to your emergency fund or savings account.")

    # General tip always shown
    recommendations.append(
        "📌 Tip: Track expenses daily — small amounts add up quickly over a month.")

    return recommendations


# STREAMLIT UI
st.set_page_config(page_title="AI Personal Finance Advisor", layout="wide")
st.title("💰 AI Personal Finance Advisor")
st.write("Predict your spending, analyze trends, and get personalized saving advice.")

# Two tabs for clean layout
tab1, tab2 = st.tabs(["Expense Predictor & Advice", "Spending Trend Analysis"])


# TAB 1 - PREDICTION + RECOMMENDATION

with tab1:
    st.subheader("Predict Your Expense")

    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Select Date")

    with col2:
        category = st.selectbox(
            "Category", ["Food", "Travel", "Shopping", "Bills"])

    # Feature engineering on user input
    input_data = pd.DataFrame({'date': [date], 'category': [category]})
    input_data['date'] = pd.to_datetime(input_data['date'])
    input_data['day'] = input_data['date'].dt.day
    input_data['month'] = input_data['date'].dt.month
    input_data = pd.get_dummies(input_data, columns=['category'])

    # Add missing columns to match training shape
    model_columns = [
        'day', 'month', 'category_Bills',
        'category_Food', 'category_Shopping', 'category_Travel'
    ]
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[model_columns]

    # Predict using RandomForestRegressor
    if st.button("Predict Expense"):
        prediction = model.predict(input_data)
        predicted_amount = round(prediction[0], 2)

        st.success(f"💸 Predicted Expense: ₹{predicted_amount}")

        # Category-wise context from real data
        st.markdown("---")
        st.subheader("How does this compare to your history?")

        cat_data = df[df['category'] == category]['amount']

        if not cat_data.empty:
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Your Avg Spend", f"₹{cat_data.mean():.0f}")
            with col4:
                st.metric("Your Max Spend", f"₹{cat_data.max():.0f}")
            with col5:
                st.metric("Predicted This Time", f"₹{predicted_amount}")

        # Personalized recommendation engine
        st.markdown("---")
        st.subheader("📊 Personalized Saving Advice")

        tips = get_recommendation(category, predicted_amount)
        for tip in tips:
            st.write(tip)


# TAB 2 - TIME-BASED & CATEGORY-WISE ANALYSIS

with tab2:
    st.subheader("Spending Trend Analysis")

    # Add month name column for readability
    df['month_name'] = pd.to_datetime(df['date']).dt.strftime('%b')
    df['month_num'] = pd.to_datetime(df['date']).dt.month

    col6, col7 = st.columns(2)

    # Chart 1 - Category-wise total spending (Seaborn barplot)
    with col6:
        st.markdown("**Category-wise Total Spending**")
        cat_summary = df.groupby('category')['amount'].sum().reset_index()

        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(
            data=cat_summary,
            x='category',
            y='amount',
            palette='Set2',
            ax=ax1
        )
        ax1.set_title("Total Spend by Category")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Total Amount (₹)")
        ax1.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig1)
        plt.close()

    # Chart 2 - Monthly total spending trend (Matplotlib line)
    with col7:
        st.markdown("**Monthly Spending Trend**")
        monthly_summary = df.groupby(['month_num', 'month_name'])[
            'amount'].sum().reset_index()
        monthly_summary = monthly_summary.sort_values('month_num')

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(
            monthly_summary['month_name'],
            monthly_summary['amount'],
            marker='o',
            color='#4CAF50',
            linewidth=2
        )
        ax2.fill_between(
            monthly_summary['month_name'],
            monthly_summary['amount'],
            alpha=0.1,
            color='#4CAF50'
        )
        ax2.set_title("Monthly Spending Trend")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Amount (₹)")
        ax2.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    # Chart 3 - Category-wise monthly breakdown (Seaborn heatmap)
    st.markdown("**Category × Month Spending Heatmap**")

    pivot = df.pivot_table(
        index='category',
        columns='month_name',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='g',
        cmap='YlOrRd',
        ax=ax3,
        linewidths=0.5
    )
    ax3.set_title("Spending Heatmap (Category vs Month)")
    st.pyplot(fig3)
    plt.close()

    # Raw data table
    st.markdown("---")
    st.subheader("Your Expense Data")
    st.dataframe(df[['date', 'category', 'amount']], use_container_width=True)
