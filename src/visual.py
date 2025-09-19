"""
Comprehensive Diabetes Analysis Visualization Suite
Combines data exploration and model performance visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DiabetesVisualizationSuite:
    """Complete visualization suite for diabetes analysis"""

    def __init__(self, data_path=None):
        # Auto-detect the correct path based on current working directory
        if data_path is None:
            import os
            if os.path.exists('data/diabetes_data.csv'):
                self.data_path = 'data/diabetes_data.csv'  # Running from project root
            elif os.path.exists('../data/diabetes_data.csv'):
                self.data_path = '../data/diabetes_data.csv'  # Running from src directory
            else:
                raise FileNotFoundError("Could not find diabetes_data.csv. Please run from project root or src directory.")
        else:
            self.data_path = data_path
        self.df = None
        self.model = None
        self.metrics = None

        # Auto-detect plots directory path
        self.plots_dir = self._get_plots_dir()

    def _get_plots_dir(self):
        """Auto-detect correct plots directory path"""
        import os
        if os.path.exists('results'):
            return 'results/plots'  # Running from project root
        else:
            return '../results/plots'  # Running from src directory

    def load_and_prepare_data(self):
        """Load and prepare data for visualization"""
        print("Loading diabetes dataset for visualization...")

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]:,} records, {self.df.shape[1]} features")

        # Apply mappings for better visualization
        self._map_columns()

        # Handle missing values
        self._handle_missing_values()

        return self.df

    def _map_columns(self):
        """Map categorical columns to descriptive labels"""
        # Map diabetes status
        self.df['Diabetes_Status'] = self.df['Diabetes_012'].map({
            0.0: 'No Diabetes', 1.0: 'Prediabetes', 2.0: 'Diabetes'
        })

        # Map physical activity
        self.df['PhysActivity'] = self.df['PhysActivity'].map({1: 'Active', 0: 'Non-Active'})

        # Map age groups
        age_mapping = {
            1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
            5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
            9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79',
            13: '80+'
        }
        self.df['Age_Group'] = self.df['Age'].map(age_mapping)

        # Map income levels
        income_mapping = {
            1: 'Less than $10,000', 2: '$10,000 - $15,000', 3: '$15,000 - $20,000',
            4: '$20,000 - $25,000', 5: '$25,000 - $35,000', 6: '$35,000 - $50,000',
            7: '$50,000 - $75,000', 8: '$75,000 or more'
        }
        self.df['Income_Level'] = self.df['Income'].map(income_mapping)

        # Map income levels (short version)
        self.df['Income_Level_Short'] = self.df['Income'].map({
            1: '<$10k', 2: '$10k-$15k', 3: '$15k-$20k', 4: '$20k-$25k',
            5: '$25k-$35k', 6: '$35k-$50k', 7: '$50k-$75k', 8: '$75k+'
        })

        # Map gender
        self.df['Sex'] = self.df['Sex'].map({0.0: 'Female', 1.0: 'Male'})

    def _handle_missing_values(self):
        """Handle missing values in critical columns"""
        critical_columns = ['Diabetes_Status', 'PhysActivity', 'Age_Group',
                           'Income_Level', 'BMI', 'MentHlth', 'Sex', 'Fruits', 'Veggies', 'PhysHlth']

        if self.df[critical_columns].isnull().any().any():
            print("Warning: Null values detected. Rows with null values will be dropped.")
            self.df = self.df.dropna(subset=critical_columns)

    def create_bar_plot(self, x_col, hue_col, title, xlabel, ylabel, ax,
                       order=None, hue_order=None, annotate=True):
        """Create customized bar plots"""
        try:
            sns.countplot(x=x_col, hue=hue_col, data=self.df, palette='Set2',
                         ax=ax, order=order, hue_order=hue_order)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Add commas to y-axis
            formatter = FuncFormatter(lambda x, pos: f'{int(x):,}')
            ax.yaxis.set_major_formatter(formatter)

            # Optionally add counts on bars
            if annotate:
                for p in ax.patches:
                    if p.get_height() > 0:
                        ax.annotate(f'{p.get_height():,}',
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='center', xytext=(0, 10),
                                   textcoords='offset points')
        except Exception as e:
            print(f"Error creating bar plot: {e}")

    def train_model_for_performance_viz(self):
        """Train model to generate performance visualizations"""
        print("Training model for performance visualization...")

        # Create binary target
        self.df['has_diabetes'] = (self.df['Diabetes_012'] > 0).astype(int)

        # Select features (use original numeric columns, not mapped ones)
        feature_cols = [
            'BMI', 'Age', 'PhysActivity', 'Income', 'HighBP', 'HighChol',
            'MentHlth', 'PhysHlth', 'Sex', 'Fruits', 'Veggies', 'GenHlth'
        ]

        # Make sure we use original numeric values for training
        # Reload original data temporarily for model training
        df_original = pd.read_csv(self.data_path)

        X = df_original[feature_cols]
        y = (df_original['Diabetes_012'] > 0).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.metrics = {
            'roc_auc': roc_auc, 'pr_auc': pr_auc,
            'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision,
            'predictions': {'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba},
            'feature_importance': feature_importance
        }

        print(f"Model trained - ROC-AUC: {roc_auc:.3f}, Sensitivity: {sensitivity:.3f}")

    def create_data_exploration_plots(self):
        """Create data exploration visualizations"""
        print("Creating data exploration visualizations...")

        # Set overall style
        plt.style.use('default')
        sns.set_palette("Set2")

        # 1. Physical Activity vs Diabetes
        plt.figure(figsize=(12, 8))
        ax1 = plt.gca()
        self.create_bar_plot(
            x_col='PhysActivity', hue_col='Diabetes_Status',
            title='Physical Activity and Diabetes Outcomes',
            xlabel='Physical Activity Status', ylabel='Population Count', ax=ax1
        )
        ax1.legend(title='Diabetes Status')
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/physical_activity_diabetes.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 2. Age Distribution by Diabetes Status
        age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                     '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

        plt.figure(figsize=(14, 8))
        ax2 = plt.gca()
        self.create_bar_plot(
            x_col='Diabetes_Status', hue_col='Age_Group',
            title='Age Distribution Across Diabetes Status',
            xlabel='Diabetes Status', ylabel='Population Count',
            ax=ax2, order=['No Diabetes', 'Prediabetes', 'Diabetes'],
            hue_order=age_order, annotate=False
        )
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/age_diabetes_distribution.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 3. BMI Heatmap by Income and Diabetes Status
        pivot_table = self.df.pivot_table(values='BMI', index='Income_Level',
                                         columns='Diabetes_Status', aggfunc='mean')

        income_order = ['Less than $10,000', '$10,000 - $15,000', '$15,000 - $20,000',
                       '$20,000 - $25,000', '$25,000 - $35,000', '$35,000 - $50,000',
                       '$50,000 - $75,000', '$75,000 or more']
        pivot_table = pivot_table.reindex(income_order)

        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm",
                   cbar_kws={'label': 'Average BMI'})
        plt.title('Average BMI by Income Level and Diabetes Status', fontsize=14, fontweight='bold')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Income Level')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/bmi_income_heatmap.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 4. Mental Health Violin Plot
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='Diabetes_Status', y='MentHlth', hue='Sex',
                      data=self.df, palette='Set2', split=True, inner="quartile")
        plt.title('Mental Health Days by Diabetes Status and Gender', fontsize=14, fontweight='bold')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Poor Mental Health Days (Last 30 Days)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/mental_health_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 5. Nutrition and Physical Health Composite Analysis
        self._create_nutrition_composite_plot()

    def _create_nutrition_composite_plot(self):
        """Create composite nutrition and physical health analysis"""
        income_order_short = ['<$10k', '$10k-$15k', '$15k-$20k', '$20k-$25k',
                             '$25k-$35k', '$35k-$50k', '$50k-$75k', '$75k+']

        df_filtered = self.df[['Fruits', 'Veggies', 'PhysHlth', 'Income_Level_Short', 'Diabetes_Status']].copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Fruits vs Physical Health
        sns.barplot(x='Fruits', y='PhysHlth', hue='Diabetes_Status',
                   palette='Set2', data=df_filtered, ax=axes[0, 0])
        axes[0, 0].set_title('Fruit Consumption vs Physical Health', fontweight='bold')
        axes[0, 0].set_xlabel('Fruit Consumption')
        axes[0, 0].set_ylabel('Average Poor Physical Health Days')
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_xticklabels(['No', 'Yes'])

        # Vegetables vs Physical Health
        sns.barplot(x='Veggies', y='PhysHlth', hue='Diabetes_Status',
                   palette='Set2', data=df_filtered, ax=axes[0, 1])
        axes[0, 1].set_title('Vegetable Consumption vs Physical Health', fontweight='bold')
        axes[0, 1].set_xlabel('Vegetable Consumption')
        axes[0, 1].set_ylabel('Average Poor Physical Health Days')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_xticklabels(['No', 'Yes'])

        # Income vs Fruit Consumption
        sns.barplot(x='Income_Level_Short', y='Fruits', hue='Diabetes_Status',
                   palette='Set2', data=df_filtered, ax=axes[1, 0], order=income_order_short)
        axes[1, 0].set_title('Income vs Fruit Consumption', fontweight='bold')
        axes[1, 0].set_xlabel('Income Level')
        axes[1, 0].set_ylabel('Average Fruit Consumption')
        axes[1, 0].tick_params(axis='x', labelsize=9, rotation=45)

        # Income vs Vegetable Consumption
        sns.barplot(x='Income_Level_Short', y='Veggies', hue='Diabetes_Status',
                   palette='Set2', data=df_filtered, ax=axes[1, 1], order=income_order_short)
        axes[1, 1].set_title('Income vs Vegetable Consumption', fontweight='bold')
        axes[1, 1].set_xlabel('Income Level')
        axes[1, 1].set_ylabel('Average Vegetable Consumption')
        axes[1, 1].tick_params(axis='x', labelsize=9, rotation=45)

        # Remove individual legends and create combined legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        plt.suptitle("Nutrition, Income, and Physical Health Across Diabetes Status",
                    fontsize=16, fontweight='bold', y=0.95)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/nutrition_composite_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

    def create_model_performance_plots(self):
        """Create model performance visualizations"""
        if self.metrics is None:
            print("Error: Model must be trained before creating performance plots")
            return

        print("Creating model performance visualizations...")

        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set style for performance plots
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(self.metrics['predictions']['y_test'],
                               self.metrics['predictions']['y_pred_proba'])

        plt.plot(fpr, tpr, linewidth=3, label=f'Model Performance (AUC = {self.metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random Guessing')
        plt.xlabel('False Positive Rate (Healthy Patients Incorrectly Flagged)', fontsize=11)
        plt.ylabel('True Positive Rate (Diabetic Patients Correctly Identified)', fontsize=11)
        plt.title('Model Ability to Separate Healthy vs. Diabetic Patients', fontsize=14, fontweight='bold')
        plt.suptitle(f'AUC = {self.metrics["roc_auc"]:.3f} (Higher is Better, Max = 1.0)', fontsize=12, y=0.02)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 2. Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(self.metrics['predictions']['y_test'],
                                                     self.metrics['predictions']['y_pred_proba'])

        plt.plot(recall, precision, linewidth=3,
                label=f'PR Curve (AUC = {self.metrics["pr_auc"]:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 3. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.metrics['predictions']['y_test'],
                             self.metrics['predictions']['y_pred'])

        # Calculate percentages for annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot_text = [[f'{cm[i,j]:,}\n({cm_percent[i,j]:.1f}%)' for j in range(cm.shape[1])] for i in range(cm.shape[0])]

        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', cbar_kws={'label': 'Patient Count'},
                   xticklabels=['Predicted: No Diabetes', 'Predicted: Has Diabetes'],
                   yticklabels=['Actually: No Diabetes', 'Actually: Has Diabetes'])
        plt.title('Correct Predictions vs. Mistakes', fontsize=14, fontweight='bold')
        plt.suptitle('Green diagonal = Correct predictions, Off-diagonal = Errors', fontsize=11, y=0.02)
        plt.ylabel('What Patients Actually Have', fontsize=12)
        plt.xlabel('What Model Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 4. Feature Importance
        plt.figure(figsize=(12, 8))
        top_features = self.metrics['feature_importance'].head(10)

        # Create more descriptive feature names
        feature_names_mapping = {
            'GenHlth': 'General Health Status',
            'HighBP': 'High Blood Pressure',
            'BMI': 'Body Mass Index (BMI)',
            'HighChol': 'High Cholesterol',
            'Age': 'Age Group',
            'PhysHlth': 'Physical Health Days',
            'Income': 'Income Level',
            'MentHlth': 'Mental Health Days',
            'PhysActivity': 'Physical Activity',
            'Sex': 'Gender',
            'Veggies': 'Vegetable Consumption',
            'Fruits': 'Fruit Consumption'
        }

        readable_features = [feature_names_mapping.get(feat, feat) for feat in top_features['feature']]
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), readable_features)
        plt.xlabel('Importance Score (Higher = More Influential)', fontsize=12)
        plt.title('Top Drivers of Diabetes Risk in This Dataset', fontsize=14, fontweight='bold')
        plt.suptitle('Features ranked by how much they influence model predictions', fontsize=11, y=0.02)
        plt.gca().invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        # 5. Model Performance Summary
        plt.figure(figsize=(12, 8))
        performance_metrics = ['ROC-AUC', 'PR-AUC', 'Sensitivity', 'Specificity', 'Precision']
        performance_values = [self.metrics['roc_auc'], self.metrics['pr_auc'],
                             self.metrics['sensitivity'], self.metrics['specificity'],
                             self.metrics['precision']]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = plt.bar(performance_metrics, performance_values, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, performance_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics Summary', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled for non-interactive mode

        print("All model performance visualizations created and saved!")
        print("\n" + "=" * 50)
        print("VISUALIZATION SUMMARY FOR RECRUITERS")
        print("=" * 50)
        print("* ROC Curve shows how well the model separates patients with vs. without diabetes")
        print("* Confusion Matrix highlights correct predictions vs. mistakes")
        print("* Feature Importance shows which health factors drive the model most")
        print("All plots saved with business-friendly labels and context!")

    def create_all_visualizations(self):
        """Create complete visualization suite"""
        print("=" * 60)
        print("DIABETES ANALYSIS VISUALIZATION SUITE")
        print("=" * 60)

        # Load data
        self.load_and_prepare_data()

        # Create data exploration plots
        print("\n--- PHASE 1: DATA EXPLORATION ---")
        self.create_data_exploration_plots()

        # Train model and create performance plots
        print("\n--- PHASE 2: MODEL PERFORMANCE ---")
        self.train_model_for_performance_viz()
        self.create_model_performance_plots()

        print("\n" + "=" * 60)
        print("VISUALIZATION SUITE COMPLETED!")
        print("=" * 60)
        print("Generated visualizations:")
        print("  Data Exploration (5 plots):")
        print("    - Physical Activity vs Diabetes")
        print("    - Age Distribution Analysis")
        print("    - BMI Income Heatmap")
        print("    - Mental Health Analysis")
        print("    - Nutrition Composite Analysis")
        print("  Model Performance (5 plots):")
        print("    - ROC Curve")
        print("    - Precision-Recall Curve")
        print("    - Confusion Matrix")
        print("    - Feature Importance")
        print("    - Performance Summary")
        print(f"\nAll plots saved to: {self.plots_dir}/")


def create_model_performance_plots(metrics):
    """Function for train.py integration"""
    viz = DiabetesVisualizationSuite()
    viz.metrics = metrics
    viz.create_model_performance_plots()


# Main execution for standalone use
if __name__ == "__main__":
    import sys

    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        matplotlib.use('TkAgg')  # Switch to interactive backend
        plt.ion()  # Turn on interactive mode
        # Re-enable plt.show() for interactive mode
        print("Running in INTERACTIVE mode - plots will display on screen")
    else:
        print("Running in BATCH mode - plots saved to files only")
        print("Use '--interactive' flag to see plots on screen")

    # Create and run the complete visualization suite
    diabetes_viz = DiabetesVisualizationSuite()
    diabetes_viz.create_all_visualizations()