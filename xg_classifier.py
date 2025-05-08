import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance
import joblib
from scipy.stats import randint, uniform
import time
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

class RecombinationRateClassifier:
    def __init__(self, output_dir="xgb_classifier_results", random_state=42):
        """Initialize the recombination rate classifier"""
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
            'chrom', 'start', 'end', 'sequence', 'sequence_id'
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
        missing_y = pd.Series(y).isna().sum()
        
        print(f"Missing values in features: {missing_X}")
        print(f"Missing values in target: {missing_y}")
        
        # Remove rows with NaN targets
        if missing_y > 0:
            print(f"Removing {missing_y} rows with NaN target values")
            valid_indices = ~pd.Series(y).isna()
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
        if isinstance(y, np.ndarray):
            assert not np.isnan(y).any(), "NaN values still present in target!"
            
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
        
        # 5. Add polynomial features for important numerical columns
        numerical_cols = X_new.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Limit to first 5 numerical columns to avoid explosion
            X_new[f'{col}_squared'] = X_new[col] ** 2
            X_new[f'{col}_cubed'] = X_new[col] ** 3
        
        print(f"Created {X_new.shape[1] - X.shape[1]} new engineered features")
        print(f"New feature matrix shape: {X_new.shape}")
        
        # Update feature names
        self.feature_names = X_new.columns.tolist()
        
        return X_new
    
    # def prepare_data(self, X, y, test_size=0.2):
    #     """Split data and prepare for training, applying SMOTE only to training data"""
    #     # Check original class distribution
    #     class_counts = pd.Series(y).value_counts()
    #     total_samples = len(y)
    #     print("\nClass distribution in full dataset:")
    #     print(class_counts)
    #     print("\nClass proportions in full dataset:")
    #     print(class_counts / total_samples)
        
    #     # First split the data before applying SMOTE
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size, random_state=self.random_state, stratify=y
    #     )
        
    #     print(f"\nTraining set shape before SMOTE: {X_train.shape}")
    #     print(f"Test set shape: {X_test.shape}")
        
    #     # Handle missing values in the training data
    #     print("\nHandling NaN values before SMOTE...")
    #     imputer = SimpleImputer(strategy='median')
    #     X_train_imputed = pd.DataFrame(
    #         imputer.fit_transform(X_train),
    #         columns=X_train.columns,
    #         index=X_train.index
    #     )
        
    #     # Verify no NaN values remain
    #     nan_count = X_train_imputed.isna().sum().sum()
    #     print(f"NaN values after imputation: {nan_count}")
        
    #     # Apply SMOTE only to the training data
    #     print("\nApplying SMOTE for class balancing on training data only...")
    #     smote = SMOTE(random_state=self.random_state)
    #     X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)
        
    #     print("\nClass distribution in training set after SMOTE balancing:")
    #     print(pd.Series(y_train_balanced).value_counts())
    #     print("\nClass proportions in training set after SMOTE balancing:")
    #     print(pd.Series(y_train_balanced).value_counts(normalize=True))
        
    #     print("\nClass distribution in test set (unchanged):")
    #     print(pd.Series(y_test).value_counts())
    #     print("\nClass proportions in test set (unchanged):")
    #     print(pd.Series(y_test).value_counts(normalize=True))
        
    #     print(f"\nTraining set shape after SMOTE: {X_train_balanced.shape}")
        
    #     # Store the splits
    #     self.X_train = X_train_balanced
    #     self.X_test = X_test
    #     self.y_train = y_train_balanced
    #     self.y_test = y_test
        
    #     return X_train_balanced, X_test, y_train_balanced, y_test
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split data and prepare for training, applying SMOTE only to training data"""
        # First split the data to keep test set with original distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTraining set shape before SMOTE: {X_train.shape}")
        print(f"Test set shape (original distribution): {X_test.shape}")
        
        # Check class distribution in training set
        train_class_counts = pd.Series(y_train).value_counts()
        print("\nClass distribution in training set before SMOTE:")
        print(train_class_counts)
        print("\nClass proportions in training set before SMOTE:")
        print(train_class_counts / len(y_train))
        
        # Check class distribution in test set
        test_class_counts = pd.Series(y_test).value_counts()
        print("\nClass distribution in test set (unchanged):")
        print(test_class_counts)
        print("\nClass proportions in test set (unchanged):")
        print(test_class_counts / len(y_test))
        
        # Handle missing values in the training data
        print("\nHandling NaN values before SMOTE...")
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Verify no NaN values remain
        nan_count = X_train_imputed.isna().sum().sum()
        print(f"NaN values after imputation: {nan_count}")
        
        # Apply SMOTE only to the training data
        print("\nApplying SMOTE for class balancing on training data only...")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)
        
        # Convert X_train_balanced to DataFrame with original column names
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        
        print("\nClass distribution in training set after SMOTE balancing:")
        print(pd.Series(y_train_balanced).value_counts())
        print("\nClass proportions in training set after SMOTE balancing:")
        print(pd.Series(y_train_balanced).value_counts(normalize=True))
        
        print(f"\nTraining set shape after SMOTE: {X_train_balanced.shape}")
        
        # Store the splits
        self.X_train = X_train_balanced
        self.X_test = X_test
        self.y_train = y_train_balanced
        self.y_test = y_test
        
        return X_train_balanced, X_test, y_train_balanced, y_test

    #     """Split data and prepare for training, with balanced real test set"""
    #     # Check original class distribution
    #     class_counts = pd.Series(y).value_counts()
    #     total_samples = len(y)
    #     print("\nClass distribution in full dataset:")
    #     print(class_counts)
    #     print("\nClass proportions in full dataset:")
    #     print(class_counts / total_samples)
        
    #     # Get indices for each class
    #     minority_class = class_counts.idxmin()
    #     majority_class = class_counts.idxmax()
        
    #     minority_indices = np.where(y == minority_class)[0]
    #     majority_indices = np.where(y == majority_class)[0]
        
    #     # Calculate how many samples to include in test set from each class
    #     # We'll use an equal number based on the minority class size and test_size
    #     minority_test_size = int(len(minority_indices) * test_size)
        
    #     # Create stratified test set with equal class representation
    #     # Take samples from minority class
    #     minority_test_indices = np.random.choice(
    #         minority_indices, 
    #         size=minority_test_size, 
    #         replace=False
    #     )
        
    #     # Take the same number of samples from majority class
    #     majority_test_indices = np.random.choice(
    #         majority_indices, 
    #         size=minority_test_size, 
    #         replace=False
    #     )
        
    #     # Combine to create balanced test set indices
    #     test_indices = np.concatenate([minority_test_indices, majority_test_indices])
        
    #     # All remaining indices go to training set
    #     train_indices = np.array([i for i in range(len(y)) if i not in test_indices])
        
    #     # Create train/test split
    #     X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    #     y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
    #     print(f"\nTraining set shape before SMOTE: {X_train.shape}")
    #     print(f"Test set shape (balanced with real examples): {X_test.shape}")
        
    #     # Check test set distribution
    #     print("\nClass distribution in balanced real test set:")
    #     print(pd.Series(y_test).value_counts())
    #     print("\nClass proportions in balanced real test set:")
    #     print(pd.Series(y_test).value_counts(normalize=True))
        
    #     # Handle missing values in the training data
    #     print("\nHandling NaN values before SMOTE...")
    #     imputer = SimpleImputer(strategy='median')
    #     X_train_imputed = pd.DataFrame(
    #         imputer.fit_transform(X_train),
    #         columns=X_train.columns,
    #         index=X_train.index
    #     )
        
    #     # Verify no NaN values remain
    #     nan_count = X_train_imputed.isna().sum().sum()
    #     print(f"NaN values after imputation: {nan_count}")
        
    #     # Apply SMOTE only to the training data
    #     print("\nApplying SMOTE for class balancing on training data only...")
    #     smote = SMOTE(random_state=self.random_state)
    #     X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)
        
    #     print("\nClass distribution in training set after SMOTE balancing:")
    #     print(pd.Series(y_train_balanced).value_counts())
    #     print("\nClass proportions in training set after SMOTE balancing:")
    #     print(pd.Series(y_train_balanced).value_counts(normalize=True))
        
    #     print(f"\nTraining set shape after SMOTE: {X_train_balanced.shape}")
        
    #     # Store the splits
    #     self.X_train = X_train_balanced
    #     self.X_test = X_test
    #     self.y_train = y_train_balanced
    #     self.y_test = y_test
        
    #     return X_train_balanced, X_test, y_train_balanced, y_test

    def train_model(self, X_train, y_train, n_iter=5, cv=3):
        """Train an XGBoost classifier with optimized parameters"""
        print("Training XGBoost classifier...")
        start_time = time.time()
        
        # Check for missing values
        if X_train.isna().any().any():
            print("Found missing values in training data. Handling them...")
            X_train = X_train.fillna(X_train.median())
        
        # Create a pipeline with scaling and XGBoost
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('xgb', xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                scale_pos_weight=1,
                tree_method='hist',
                enable_categorical=False
            ))
        ])
        
        # Define parameter grid for optimization
        param_dist = {
            'xgb__max_depth': randint(3, 7),
            'xgb__learning_rate': uniform(0.05, 0.2),
            'xgb__subsample': uniform(0.7, 0.3),
            'xgb__colsample_bytree': uniform(0.7, 0.3),
            'xgb__min_child_weight': randint(1, 5),
            'xgb__gamma': uniform(0, 0.3),
            'xgb__reg_alpha': uniform(0, 0.5),
            'xgb__reg_lambda': uniform(0, 1)
        }
        
        # Perform random search
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        # Print best parameters
        print("\nBest parameters found:")
        print(random_search.best_params_)
        
        # Print class distribution in training set
        print("\nClass distribution in training set:")
        print(pd.Series(y_train).value_counts())
        print("\nClass proportions in training set:")
        print(pd.Series(y_train).value_counts(normalize=True))
        
        # Print training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        print(f"Unique predicted classes: {np.unique(y_pred)}")
        print(f"Unique actual classes: {np.unique(y_test)}")
        
        # Get probability predictions
        proba = self.model.predict_proba(X_test)
        print(f"Probability shape: {proba.shape}")
        print(f"Probability values: {proba[:5]}")  # Print first 5 predictions
        
        # Get probability of positive class
        if proba.shape[1] == 1:
            y_pred_proba = proba.ravel()
        else:
            y_pred_proba = proba[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Compute and print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("----------------")
        print("True Negative (TN):", cm[0, 0])
        print("False Positive (FP):", cm[0, 1])
        print("False Negative (FN):", cm[1, 0])
        print("True Positive (TP):", cm[1, 1])
        print("\nConfusion Matrix (as percentage of total):")
        cm_percent = cm / cm.sum() * 100
        print("True Negative (TN): {:.2f}%".format(cm_percent[0, 0]))
        print("False Positive (FP): {:.2f}%".format(cm_percent[0, 1]))
        print("False Negative (FN): {:.2f}%".format(cm_percent[1, 0]))
        print("True Positive (TP): {:.2f}%".format(cm_percent[1, 1]))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
        
        # Save metrics to file
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        }
        pd.DataFrame([metrics]).to_csv(os.path.join(self.output_dir, 'model_metrics.csv'), index=False)
        
        return metrics
    
    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using multiple methods"""
        print("\nAnalyzing feature importance...")
        
        # Get the XGBoost model from the pipeline
        xgb_model = self.model.named_steps['xgb']
        
        # Method 1: Built-in feature importance
        importance = xgb_model.feature_importances_
        feature_names = self.feature_names
        
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
        
        # Create plot
        plt.figure(figsize=(12, 10))
        top_n = min(30, len(importance_df))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Features by Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()
        
        # Store for later use
        self.feature_importances = importance_df
        
        # Method 2: Try permutation importance (on a smaller subset)
        try:
            print("\nCalculating permutation importance (this may take a while)...")
            # Use a subsample for computational efficiency
            max_samples = min(1000, self.X_test.shape[0])
            sample_indices = np.random.choice(
                self.X_test.shape[0], max_samples, replace=False
            )
            X_sample = self.X_test.iloc[sample_indices]
            y_sample = self.y_test.iloc[sample_indices]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X_sample, y_sample, 
                n_repeats=5, random_state=self.random_state,
                scoring='roc_auc'
            )
            
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
            
            # Create plot
            plt.figure(figsize=(12, 10))
            top_n = min(30, len(perm_df))
            sns.barplot(x='Importance', y='Feature', data=perm_df.head(top_n))
            plt.title(f'Top {top_n} Features by Permutation Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'permutation_importance.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error calculating permutation importance: {str(e)}")
        
        return self.feature_importances
    
    def save_model(self, filename='xgb_classifier.pkl'):
        """Save the trained model to disk"""
        if self.model is not None:
            joblib.dump(self.model, os.path.join(self.output_dir, filename))
            print(f"Model saved to {os.path.join(self.output_dir, filename)}")
    
    def run_pipeline(self, file_path, test_size=0.2, n_iter=20, cv=5):
        """Run the complete analysis pipeline"""
        # 1. Load and prepare data
        X, y, df = self.load_data(file_path)
        
        # 2. Engineer features
        X_engineered = self.engineer_features(X, y)
        
        # 3. Split data
        X_train, X_test, y_train, y_test = self.prepare_data(X_engineered, y, test_size)
        
        # 4. Train model
        self.train_model(X_train, y_train, n_iter, cv)
        
        # 5. Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # 6. Analyze feature importance
        importance_df = self.analyze_feature_importance(X_engineered, y)
        
        # 7. Save model
        self.save_model()
        
        print("\nAnalysis complete!")
        print(f"All results saved to: {self.output_dir}")
        
        return metrics, importance_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train an optimized XGBoost classifier for binary recombination rate prediction")
    parser.add_argument("--input", "-i", required=True, help="Path to input TSV file")
    parser.add_argument("--output", "-o", default="xgb_classifier_results", help="Output directory")
    parser.add_argument("--test-size", "-t", type=float, default=0.2, help="Test set size (proportion)")
    parser.add_argument("--n-iter", "-n", type=int, default=20, help="Number of hyperparameter combinations to try")
    parser.add_argument("--cv", "-c", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Run the pipeline
    classifier = RecombinationRateClassifier(output_dir=args.output, random_state=args.seed)
    classifier.run_pipeline(
        file_path=args.input,
        test_size=args.test_size,
        n_iter=args.n_iter,
        cv=args.cv
    ) 