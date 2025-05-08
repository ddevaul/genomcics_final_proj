#!/usr/bin/env python3
"""
Advanced Random Forest Model for Recombination Rate Prediction
with enhanced preprocessing, optimization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
from sklearn.inspection import permutation_importance
import joblib
from scipy.stats import randint, uniform
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class RecombinationRatePredictor:
    def __init__(self, output_dir="rf_improved_results", random_state=42):
        """Initialize the recombination rate predictor"""
        self.output_dir = output_dir
        self.random_state = random_state
        self.model = None
        self.feature_importances = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, file_path):
        """Load and preprocess the data"""
        print("Loading data...")
        df = pd.read_csv(file_path, sep='\t')
        print(f"Dataset shape: {df.shape}")
        
        # Define columns to exclude
        exclude_columns = [
            'chrom', 'start', 'end', 'sequence', 'sequence_id',
        ]
        exclude_columns = [col for col in exclude_columns if col in df.columns]
        
        # Create features and target
        X = df.drop(['recomb_rate'] + exclude_columns, axis=1)
        y = df['recomb_rate']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Handle missing values
        X, y = self._handle_missing_values(X, y)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X, y, df
    
    def _handle_missing_values(self, X, y):
        """Thoroughly handle missing and invalid values"""
        # Check for NaN values
        missing_X = X.isna().sum().sum()
        missing_y = y.isna().sum()
        
        print(f"Missing values in features: {missing_X}")
        print(f"Missing values in target: {missing_y}")
        
        # Remove rows with NaN targets
        if missing_y > 0:
            print(f"Removing {missing_y} rows with NaN target values")
            valid_indices = ~y.isna()
            if isinstance(X, pd.DataFrame):
                X = X[valid_indices]
            else:
                X = X[valid_indices.values]
            y = y[valid_indices]
        
        # Handle NaN and infinite values in features
        if isinstance(X, pd.DataFrame):
            if X.isna().sum().sum() > 0:
                print("Filling NaN values in features with median values")
                X = X.fillna(X.median())
            
            if np.isinf(X.values).any():
                print("Replacing infinite values in features")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.median())
        
        # Final verification
        if isinstance(X, pd.DataFrame):
            assert not np.isnan(X.values).any(), "NaN values still present in features!"
        if isinstance(y, pd.Series):
            assert not np.isnan(y.values).any(), "NaN values still present in target!"
            
        return X, y
    
    def engineer_features(self, X, y=None):
        """Create engineered features to improve model performance"""
        print("Engineering additional features...")
        
        # Copy the dataframe to avoid modifying the original
        X_new = X.copy()
        
        # 1. Group k-mer features by their prefix and calculate statistics
        kmer_groups = {
            'k1': [col for col in X.columns if col.endswith('_k1')],
            'k2': [col for col in X.columns if col.endswith('_k2')],
            'k3': [col for col in X.columns if col.endswith('_k3')]
        }
        
        # Calculate statistics for each group
        for group, cols in kmer_groups.items():
            if cols:
                X_new[f'{group}_mean'] = X[cols].mean(axis=1)
                X_new[f'{group}_std'] = X[cols].std(axis=1)
                X_new[f'{group}_max'] = X[cols].max(axis=1)
                X_new[f'{group}_min'] = X[cols].min(axis=1)
                X_new[f'{group}_range'] = X_new[f'{group}_max'] - X_new[f'{group}_min']
        
        # 2. Create nucleotide ratio features
        nucleotides = ['A', 'C', 'G', 'T']
        for n1 in nucleotides:
            for n2 in nucleotides:
                # Create ratio features for k1
                k1_cols = [col for col in X.columns if col.startswith(f'{n1}{n2}_k1')]
                if k1_cols:
                    X_new[f'{n1}{n2}_ratio_k1'] = X[k1_cols].sum(axis=1) / X[[col for col in X.columns if col.endswith('_k1')]].sum(axis=1)
        
        # 3. Calculate GC content-related features
        gc_cols_k1 = [col for col in X.columns if any(col.startswith(f'{n}') for n in ['G', 'C']) and col.endswith('_k1')]
        at_cols_k1 = [col for col in X.columns if any(col.startswith(f'{n}') for n in ['A', 'T']) and col.endswith('_k1')]
        
        if gc_cols_k1 and at_cols_k1:
            X_new['gc_content_k1'] = X[gc_cols_k1].sum(axis=1) / (X[gc_cols_k1].sum(axis=1) + X[at_cols_k1].sum(axis=1))
        
        # 4. Create interaction features with strength (if present)
        if 'strength' in X.columns:
            for group in kmer_groups:
                if kmer_groups[group]:
                    X_new[f'strength_{group}_interaction'] = X['strength'] * X_new[f'{group}_mean']
        
        # 5. Create polynomial features for strongest predictors
        # This will be done after initial feature importance analysis
        
        print(f"Created {X_new.shape[1] - X.shape[1]} new engineered features")
        print(f"New feature matrix shape: {X_new.shape}")
        
        # Update feature names
        self.feature_names = X_new.columns.tolist()
        
        return X_new
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split data and prepare for training"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Store the splits
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, n_iter=20, cv=5):
        """Train an optimized random forest model with overfitting prevention"""
        print("Training optimized random forest model...")
        start_time = time.time()
        
        # Define the parameter grid for random search with anti-overfitting parameters
        param_dist = {
            'n_estimators': randint(200, 500),  # More trees for stability
            'max_depth': [None] + list(randint(5, 15).rvs(5)),  # Shallower trees
            'min_samples_split': randint(5, 20),  # Higher minimum samples for splits
            'min_samples_leaf': randint(3, 10),  # Higher minimum samples for leaves
            'max_features': ['sqrt', 'log2', 0.3, 0.5],  # More feature restriction
            'bootstrap': [True],  # Always use bootstrap
            'max_samples': uniform(0.7, 0.3),  # Higher minimum sample size
            'min_impurity_decrease': uniform(0.0, 0.1)  # Prevent splits that don't improve much
        }
        
        # Create a pipeline with scaling and random forest
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('rf', RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True  # Enable out-of-bag scoring
            ))
        ])
        
        # Wrap with target transformation using PowerTransformer
        from sklearn.preprocessing import PowerTransformer
        model = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(method='yeo-johnson')
        )
        
        # Add progress bar for model fitting
        print("Fitting model...")
        with tqdm(total=1, desc="Model training") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        
        # Get the random forest from the pipeline
        rf_model = model.regressor_.steps[-1][1]
        
        # Print OOB score if available
        if hasattr(rf_model, 'oob_score_'):
            print(f"\nOut-of-bag R² score: {rf_model.oob_score_:.4f}")
        
        self.model = model
        
        # Print training time
        training_time = time.time() - start_time
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate baseline prediction (mean of y_test)
        y_baseline = np.full_like(y_test, y_test.mean())
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mse_baseline = mean_squared_error(y_test, y_baseline)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print diagnostic information
        print("\nDiagnostic Information:")
        print(f"Mean of actual values: {y_test.mean():.4f}")
        print(f"Mean of predicted values: {y_pred.mean():.4f}")
        print(f"MSE of model: {mse:.4f}")
        print(f"MSE of baseline (mean): {mse_baseline:.4f}")
        print(f"Model MSE is {'worse' if mse > mse_baseline else 'better'} than baseline")
        
        # Calculate Spearman correlation
        spearman_corr, _ = spearmanr(y_test, y_pred)
        
        # Calculate F-statistic and p-value for R² significance
        n = len(y_test)  # number of samples
        p = X_test.shape[1]  # number of features
        print("p",p)
        df1 = p  # degrees of freedom for model
        df2 = n - p - 1  # degrees of freedom for error
        
        # Calculate F-statistic
        if r2 < 0:
            f_stat = 0  # F-statistic is not defined for negative R²
            p_value = 1.0
        else:
            f_stat = (r2 / df1) / ((1 - r2) / df2)
            # Calculate p-value using F-distribution
            from scipy.stats import f
            p_value = 1 - f.cdf(f_stat, df1, df2)
        
        # Perform permutation test for R²
        print("\nPerforming permutation test for R² significance...")
        n_permutations = 1000  # Number of permutations
        permuted_r2s = []
        
        with tqdm(total=n_permutations, desc="Permutation test") as pbar:
            for _ in range(n_permutations):
                # Permute the target variable
                y_permuted = np.random.permutation(y_test)
                # Calculate R² for permuted data
                r2_permuted = r2_score(y_permuted, y_pred)
                permuted_r2s.append(r2_permuted)
                pbar.update(1)
        
        # Calculate permutation p-value
        perm_p_value = (np.sum(np.array(permuted_r2s) >= r2) + 1) / (n_permutations + 1)
        
        # Plot permutation test results
        plt.figure(figsize=(10, 6))
        plt.hist(permuted_r2s, bins=50, alpha=0.7, label='Permuted R² values')
        plt.axvline(x=r2, color='r', linestyle='--', label=f'Actual R² = {r2:.4f}')
        plt.xlabel('R² Score')
        plt.ylabel('Frequency')
        plt.title(f'Permutation Test for R² Significance\n(p-value = {perm_p_value:.4f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'r2_permutation_test.png'))
        plt.close()
        
        # Create diagnostic plots
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(131)
        plt.scatter(y_test, y_pred, alpha=0.3)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        # Plot 2: Residuals
        plt.subplot(132)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Plot 3: Distribution comparison
        plt.subplot(133)
        sns.kdeplot(y_test, label='Actual', color='blue')
        sns.kdeplot(y_pred, label='Predicted', color='red')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'diagnostic_plots.png'))
        plt.close()
        
        # Print metrics
        print(f"\nMean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"R² F-statistic: {f_stat:.4f}")
        print(f"R² F-test p-value: {p_value:.4f}")
        print(f"R² Permutation p-value: {perm_p_value:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(y_test, y_pred, alpha=0.3)
        
        # Add identity line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(y_test.values.reshape(-1, 1), y_pred)
        plt.plot(
            [min_val, max_val], 
            [lr.predict([[min_val]])[0], lr.predict([[max_val]])[0]], 
            'g-'
        )
        
        # Add labels and title
        plt.xlabel('Actual Recombination Rate')
        plt.ylabel('Predicted Recombination Rate')
        plt.title(f'Actual vs Predicted Recombination Rate\n(R² = {r2:.4f}, p = {perm_p_value:.4f})')
        
        # Add text box with metrics
        textstr = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}\nR² p-value = {perm_p_value:.4f}\nSpearman = {spearman_corr:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', bbox=props)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        # Save metrics to file
        metrics = {
            'MSE': mse,
            'MSE_baseline': mse_baseline,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'R2_F_statistic': f_stat,
            'R2_F_test_p_value': p_value,
            'R2_permutation_p_value': perm_p_value,
            'Spearman': spearman_corr
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, 'model_metrics.csv'), index=False)
        
        return metrics
    
    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using multiple methods"""
        print("\nAnalyzing feature importance...")
        start_time = time.time()
        
        # Get the raw random forest from the pipeline
        if hasattr(self.model, 'regressor_'):
            if hasattr(self.model.regressor_, 'steps'):
                rf_model = self.model.regressor_.steps[-1][1]
            else:
                rf_model = self.model.regressor_
        else:
            rf_model = self.model
        
        # Method 1: Built-in feature importance
        if hasattr(rf_model, 'feature_importances_'):
            print("Calculating built-in feature importance...")
            with tqdm(total=1, desc="Feature importance") as pbar:
                importance = rf_model.feature_importances_
                feature_names = self.feature_names
                pbar.update(1)
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Save to CSV
            importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            
            # Print top features
            print("\nTop 15 most important features:")
            print(importance_df.head(15))
            
            # Store for later use
            self.feature_importances = importance_df
        
        # Method 2: Try permutation importance (on a smaller subset)
        try:
            print("\nCalculating permutation importance...")
            # Use a subsample for computational efficiency
            max_samples = min(1000, self.X_test.shape[0])
            sample_indices = np.random.choice(
                self.X_test.shape[0], max_samples, replace=False
            )
            X_sample = self.X_test.iloc[sample_indices]
            y_sample = self.y_test.iloc[sample_indices]
            
            # Calculate permutation importance with progress bar
            with tqdm(total=5, desc="Permutation importance") as pbar:
                perm_importance = permutation_importance(
                    self.model, X_sample, y_sample, 
                    n_repeats=5, random_state=self.random_state
                )
                pbar.update(5)
            
            # Create dataframe
            perm_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': perm_importance.importances_mean
            })
            perm_df = perm_df.sort_values('Importance', ascending=False)
            
            # Save to CSV
            perm_df.to_csv(os.path.join(self.output_dir, 'permutation_importance.csv'), index=False)
            
            # Print top features
            print("\nTop 15 features by permutation importance:")
            print(perm_df.head(15))
            
        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}")
        
        # Analyze feature groups
        self._analyze_feature_groups()
        
        print(f"Feature importance analysis completed in {time.time() - start_time:.2f} seconds")
        return self.feature_importances
    
    def _analyze_feature_groups(self):
        """Analyze importance by feature groups"""
        if self.feature_importances is None:
            print("Feature importance not calculated yet")
            return
        
        # Define feature groups
        kmer_patterns = {
            'k1': [feat for feat in self.feature_importances['Feature'] if feat.endswith('_k1')],
            'k2': [feat for feat in self.feature_importances['Feature'] if feat.endswith('_k2')],
            'k3': [feat for feat in self.feature_importances['Feature'] if feat.endswith('_k3')]
        }
        
        # Add engineered feature groups
        kmer_patterns['k1_derived'] = [feat for feat in self.feature_importances['Feature'] 
                                       if any(s in feat for s in ['k1_mean', 'k1_std', 'k1_max', 'ratio_k1', 'gc_content_k1'])]
        kmer_patterns['k2_derived'] = [feat for feat in self.feature_importances['Feature'] 
                                       if any(s in feat for s in ['k2_mean', 'k2_std', 'k2_max'])]
        kmer_patterns['k3_derived'] = [feat for feat in self.feature_importances['Feature'] 
                                       if any(s in feat for s in ['k3_mean', 'k3_std', 'k3_max'])]
        
        # Add strength as its own group if present
        if 'strength' in self.feature_importances['Feature'].values:
            kmer_patterns['strength'] = ['strength']
        
        # Add strength interactions as a group
        strength_interactions = [feat for feat in self.feature_importances['Feature'] 
                                if 'strength_' in feat and '_interaction' in feat]
        if strength_interactions:
            kmer_patterns['strength_interactions'] = strength_interactions
        
        # Calculate importance for each group
        group_importance = {}
        for group, features in kmer_patterns.items():
            if features:  # Only if the group has features
                mask = self.feature_importances['Feature'].isin(features)
                if any(mask):
                    group_importance[group] = self.feature_importances.loc[mask, 'Importance'].sum()
                    group_importance[f'{group}_count'] = len(features)
                    if len(features) > 0:
                        group_importance[f'{group}_avg'] = group_importance[group] / len(features)
        
        # Create dataframe
        group_df = pd.DataFrame({
            'Group': list(group_importance.keys()),
            'Value': list(group_importance.values())
        })
        
        # Separate the different types of metrics
        total_importance = group_df[~group_df['Group'].str.contains('_count|_avg')]
        count = group_df[group_df['Group'].str.contains('_count')]
        avg_importance = group_df[group_df['Group'].str.contains('_avg')]
        
        # Sort and rename
        total_importance = total_importance.sort_values('Value', ascending=False)
        total_importance = total_importance.rename(columns={'Value': 'Total_Importance'})
        
        # Adjust group names for count and average
        count['Group'] = count['Group'].str.replace('_count', '')
        count = count.rename(columns={'Value': 'Feature_Count'})
        
        avg_importance['Group'] = avg_importance['Group'].str.replace('_avg', '')
        avg_importance = avg_importance.rename(columns={'Value': 'Average_Importance'})
        
        # Save to CSV
        total_importance.to_csv(os.path.join(self.output_dir, 'group_importance.csv'), index=False)
        
        # Print results
        print("\nImportance by feature groups:")
        print(total_importance)
        
        # Plot group importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total_Importance', y='Group', data=total_importance)
        plt.title('Feature Importance by Groups')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'group_importance.png'))
        plt.close()
        
        # Plot average importance per feature
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Average_Importance', y='Group', data=avg_importance)
        plt.title('Average Feature Importance by Groups')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'average_group_importance.png'))
        plt.close()
        
        return total_importance
    
    def save_model(self, filename='rf_model.pkl'):
        """Save the trained model to disk"""
        if self.model is not None:
            joblib.dump(self.model, os.path.join(self.output_dir, filename))
            print(f"Model saved to {os.path.join(self.output_dir, filename)}")
    
    def run_pipeline(self, file_path, test_size=0.2, n_iter=20, cv=5):
        """Run the complete analysis pipeline"""
        total_start_time = time.time()
        
        print("\nStarting pipeline execution...")
        print("="*50)
        
        # 1. Load and prepare data
        print("\n1. Loading and preparing data...")
        start_time = time.time()
        X, y, df = self.load_data(file_path)
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        
        # 2. Engineer features
        print("\n2. Engineering features...")
        start_time = time.time()
        X_engineered = self.engineer_features(X, y)
        print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
        
        # 3. Split data
        print("\n3. Splitting data...")
        start_time = time.time()
        X_train, X_test, y_train, y_test = self.prepare_data(X_engineered, y, test_size)
        print(f"Data splitting completed in {time.time() - start_time:.2f} seconds")
        
        # 4. Train model
        print("\n4. Training model...")
        start_time = time.time()
        self.train_model(X_train, y_train, n_iter, cv)
        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        # 5. Evaluate model
        print("\n5. Evaluating model...")
        start_time = time.time()
        metrics = self.evaluate_model(X_test, y_test)
        print(f"Model evaluation completed in {time.time() - start_time:.2f} seconds")
        
        # 6. Analyze feature importance
        print("\n6. Analyzing feature importance...")
        start_time = time.time()
        importance_df = self.analyze_feature_importance(X_engineered, y)
        print(f"Feature importance analysis completed in {time.time() - start_time:.2f} seconds")
        
        # 7. Save model
        print("\n7. Saving model...")
        start_time = time.time()
        self.save_model()
        print(f"Model saving completed in {time.time() - start_time:.2f} seconds")
        
        total_time = time.time() - total_start_time
        print("\n" + "="*50)
        print(f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"All results saved to: {self.output_dir}")
        
        return metrics, importance_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train an optimized Random Forest model for recombination rate prediction")
    parser.add_argument("--input", "-i", required=True, help="Path to input TSV file")
    parser.add_argument("--output", "-o", default="rf_improved_results", help="Output directory")
    parser.add_argument("--test-size", "-t", type=float, default=0.2, help="Test set size (proportion)")
    parser.add_argument("--n-iter", "-n", type=int, default=20, help="Number of hyperparameter combinations to try")
    parser.add_argument("--cv", "-c", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Run the pipeline
    predictor = RecombinationRatePredictor(output_dir=args.output, random_state=args.seed)
    predictor.run_pipeline(
        file_path=args.input,
        test_size=args.test_size,
        n_iter=args.n_iter,
        cv=args.cv
    )