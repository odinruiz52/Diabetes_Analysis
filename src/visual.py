import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Load the data once at the start
data_path = './data/diabetes_data.csv'

# Load only necessary columns
columns_to_load = ['Diabetes_012', 'PhysActivity', 'Age', 'Income', 'BMI', 'MentHlth', 'Sex', 'Fruits', 'Veggies', 'PhysHlth']
df = pd.read_csv(data_path, usecols=columns_to_load)

# Perform all necessary mappings immediately after loading the data
def map_columns(df):
    # Map the Diabetes_012 column to descriptive labels
    df['Diabetes_Status'] = df['Diabetes_012'].map({0.0: 'No Diabetes', 1.0: 'Prediabetes', 2.0: 'Diabetes'})
    
    # Map 1 and 0 to 'Yes' and 'No' for Physical Activity
    df['PhysActivity'] = df['PhysActivity'].map({1: 'Active', 0: 'Non-Active'})

    # Correct age group mapping
    age_mapping = {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
        5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
        9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79',
        13: '80+'
    }
    df['Age_Group'] = df['Age'].map(age_mapping)
    
    # Map income levels to a readable format
    income_mapping = {
        1: 'Less than $10,000', 2: '$10,000 - $15,000', 3: '$15,000 - $20,000',
        4: '$20,000 - $25,000', 5: '$25,000 - $35,000', 6: '$35,000 - $50,000',
        7: '$50,000 - $75,000', 8: '$75,000 or more'
    }
    df['Income_Level'] = df['Income'].map(income_mapping)
    
    # Map the Sex column to descriptive labels
    df['Sex'] = df['Sex'].map({0.0: 'Female', 1.0: 'Male'})
    
    return df

# Apply the mappings
df = map_columns(df)

# Check for null values in critical columns and handle them
critical_columns = ['Diabetes_Status', 'PhysActivity', 'Age_Group', 'Income_Level', 'BMI', 'MentHlth', 'Sex', 'Fruits', 'Veggies', 'PhysHlth']
if df[critical_columns].isnull().any().any():
    print("Warning: Null values detected in critical columns. Rows with null values will be dropped.")
    df = df.dropna(subset=critical_columns, inplace=True)

# Function to create bar plots with an optional parameter to disable annotations
def create_bar_plot(df, x_col, hue_col, title, xlabel, ylabel, ax, order=None, hue_order=None, annotate=True):
    try:
        sns.countplot(x=x_col, hue=hue_col, data=df, palette='Set2', ax=ax, order=order, hue_order=hue_order)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add commas to the y-axis
        formatter = FuncFormatter(lambda x, pos: f'{int(x):,}')
        ax.yaxis.set_major_formatter(formatter)
        
        # Optionally add counts on top of each bar
        if annotate:
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():,}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 10), 
                                textcoords='offset points')
    except Exception as e:
        print(f"Error occurred while creating bar plot: {e}")

# Plot 1: Physical Activity vs. Diabetes Outcome (with annotations)
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
create_bar_plot(df, x_col='PhysActivity', hue_col='Diabetes_Status', 
                title='The Relationship Between Exercise and the Incidence of Diabetes', 
                xlabel='Physical Activity (Active/Non-Active)', ylabel='Amount of People Within Different Diabetes Groups', ax=ax1)
ax1.legend(title='Diabetes Status')
plt.show()

# Ensure the correct order of age groups
age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
             '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

# Plot 2: Age Distribution by Diabetes Status (without annotations)
plt.figure(figsize=(12, 8))
ax2 = plt.gca()
create_bar_plot(df, x_col='Diabetes_Status', hue_col='Age_Group', 
                title='Count of Age Groups by Diabetes Status', 
                xlabel='Diabetes Status', ylabel='Population Count', 
                ax=ax2, order=['No Diabetes', 'Prediabetes', 'Diabetes'], hue_order=age_order, annotate=False)
plt.show()

# Pivot the data for heatmap
pivot_table = df.pivot_table(values='BMI', index='Income_Level', 
                             columns='Diabetes_Status', aggfunc='mean')

# Reorder the income levels for the heatmap
income_order = ['Less than $10,000', '$10,000 - $15,000', '$15,000 - $20,000',
                '$20,000 - $25,000', '$25,000 - $35,000', '$35,000 - $50,000', 
                '$50,000 - $75,000', '$75,000 or more']
pivot_table = pivot_table.reindex(income_order)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Average BMI'})
plt.title('Average BMI Based by Income Level and Diabetes Status')
plt.xlabel('Diabetes Status')
plt.ylabel('Income Level')
plt.xticks(rotation=45, ha='right')
plt.show()

# Function to create violin plot
def create_violin_plot(df, x_col, y_col, hue_col, title, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='Set2', split=True, inner="quartile")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Plot 3: Mental Health Days by Diabetes Status and Gender
create_violin_plot(df, x_col='Diabetes_Status', y_col='MentHlth', 
                   hue_col='Sex', title='Distribution of Mental Health Days Based by Diabetes Status and Gender', 
                   xlabel='Diabetes Status', ylabel='Reported Poor Mental Health Days (Last 30 Days)')

# Simplify income mapping for last plot
df['Income_Level_Short'] = df['Income'].map({
    1: '<$10k', 2: '$10k-$15k', 3: '$15k-$20k', 4: '$20k-$25k', 
    5: '$25k-$35k', 6: '$35k-$50k', 7: '$50k-$75k', 8: '$75k+'
})

# Ensure the correct order of income levels
income_order_short = ['<$10k', '$10k-$15k', '$15k-$20k', '$20k-$25k', 
                      '$25k-$35k', '$35k-$50k', '$50k-$75k', '$75k+']

# Create the bar plot for Physical Health and Nutrition
def create_composite_bar_plots(df):
    # Select only the relevant columns for the custom plot
    df_filtered = df[['Fruits', 'Veggies', 'PhysHlth', 'Income_Level_Short', 'Diabetes_Status']].copy()

    # Setup the grid for subplots with adjusted figure size
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Fruits vs Physical Health
    sns.barplot(x='Fruits', y='PhysHlth', hue='Diabetes_Status', palette='Set2', data=df_filtered, ax=axes[0, 0])
    axes[0, 0].set_title('Fruits vs Physical Health')
    axes[0, 0].set_xlabel('Fruit Consumption')
    axes[0, 0].set_ylabel('Average Healthy Physical Days')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['No', 'Yes'])

    # Plot 2: Veggies vs Physical Health
    sns.barplot(x='Veggies', y='PhysHlth', hue='Diabetes_Status', palette='Set2', data=df_filtered, ax=axes[0, 1])
    axes[0, 1].set_title('Veggies vs Physical Health')
    axes[0, 1].set_xlabel('Veggie Consumption')
    axes[0, 1].set_ylabel('Average Healthy Physical Days')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['No', 'Yes'])

    # Plot 3: Income vs Fruit Consumption
    sns.barplot(x='Income_Level_Short', y='Fruits', hue='Diabetes_Status', palette='Set2', data=df_filtered, ax=axes[1, 0], order=income_order_short)
    axes[1, 0].set_title('Income vs Fruit Consumption')
    axes[1, 0].set_xlabel('Income Level')
    axes[1, 0].set_ylabel('Average Fruit Consumption')
    axes[1, 0].tick_params(axis='x', labelsize=8)

    # Plot 4: Income vs Veggie Consumption
    sns.barplot(x='Income_Level_Short', y='Veggies', hue='Diabetes_Status', palette='Set2', data=df_filtered, ax=axes[1, 1], order=income_order_short)
    axes[1, 1].set_title('Income vs Veggie Consumption')
    axes[1, 1].set_xlabel('Income Level')
    axes[1, 1].set_ylabel('Average Veggie Consumption')
    axes[1, 1].tick_params(axis='x', labelsize=8)

    # Adjust layout and move legend outside
    plt.tight_layout()
    plt.suptitle("Impact of Nutrition and Income on Physical Health Across Diabetes Status", y=1.03)

    # Get the handles and labels for the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Remove individual legends
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Place the combined legend outside the plot
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.show()

# Create the composite bar plots
create_composite_bar_plots(df)