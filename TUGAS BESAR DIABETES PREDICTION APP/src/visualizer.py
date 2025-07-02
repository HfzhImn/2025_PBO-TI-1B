import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

def plot_target_distribution(y, target_mapping):
    counts = y.value_counts().sort_index()
    labels = [target_mapping[k] for k in counts.index]
    colors = ['#ffe5c2', '#ffba66', '#FF8303']
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'color':'#FF8303', 'fontsize': 13})
    st.pyplot(fig)

def plot_bmi_histogram_with_user(df, user_bmi):
    fig, ax = plt.subplots()
    sns.histplot(df['BMI'], bins=30, color="#FF8303", ax=ax)
    ax.axvline(user_bmi, color='red', linestyle='--', linewidth=2, label='BMI Anda')
    ax.set_title("Distribusi BMI (Garis Merah = Input Anda)")
    ax.legend()
    st.pyplot(fig)

def plot_corr_with_target(df):
    corr = df.corr()['Diabetes_012'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,8))
    sns.barplot(y=corr.index, x=corr.values, palette='Oranges_r', ax=ax)
    ax.set_title("Korelasi Fitur dengan Target")
    st.pyplot(fig)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(7,4))
    sns.barplot(x=importances[indices][:10], y=[feature_names[i] for i in indices][:10], palette='Oranges_r')
    plt.title("Top 10 Feature Importance")
    st.pyplot(plt.gcf())

def plot_confusion_matrix(cm, target_names):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    st.pyplot(fig)
