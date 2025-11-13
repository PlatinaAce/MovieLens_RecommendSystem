"""
MovieLens Collaborative Filtering Recommender System
====================================================
A comprehensive comparison of various collaborative filtering algorithms

Included Algorithms:
- Item-Item kNN (Item-based k-Nearest Neighbors)
- User-User kNN (User-based k-Nearest Neighbors)
- Slope-One
- Bias-SVD (Biased Singular Value Decomposition)
- NMF (Non-negative Matrix Factorization)
- ALS (Alternating Least Squares)

Features:
- Automated hyperparameter tuning with cross-validation
- Comprehensive evaluation metrics (RMSE, MAE)
- Result visualization with matplotlib
- Top-10 movie recommendations for users
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# Surprise library for collaborative filtering
from surprise import Dataset, Reader, KNNBasic, SlopeOne, SVD, NMF as SurpriseNMF
from surprise.model_selection import GridSearchCV
from surprise import PredictionImpossible

# Implicit library for matrix factorization
import implicit
from implicit.als import AlternatingLeastSquares


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration settings for the experiment"""
    
    FAST_MODE: bool = False          # True: quick experiment, False: comprehensive grid search
    TEST_SIZE: float = 0.2          # Test set ratio (20%)
    VALID_SIZE: float = 0.1         # Validation set ratio (10%)
    RANDOM_STATE: int = 42          # Random seed for reproducibility
    
    MIN_RATING: float = 0.5         # Minimum rating value
    MAX_RATING: float = 5.0         # Maximum rating value
    
    CV_FOLDS: int = 3               # Number of cross-validation folds
    N_JOBS: int = -1                # Number of CPU cores (-1: use all available cores)


config = ExperimentConfig()


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def transform_data(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform pivot table to long format for model training
    
    Args:
        pivot_df: User x Movie pivot table with ratings
        
    Returns:
        DataFrame in long format with columns (userId, movieId, rating)
    """
    start_time = time.time()
    print("\n[Step 1] Transforming data...")
    
    # Convert pivot table to long format using stack operation
    stacked = pivot_df.stack()
    df = stacked.reset_index()
    df.columns = ['userId', 'movieId', 'rating']
    
    elapsed = time.time() - start_time
    print(f"  Completed: {len(df):,} ratings ({elapsed:.1f}s)")
    
    return df


def split_data(
    ratings_df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    valid_size: float = config.VALID_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    
    Args:
        ratings_df: Complete rating data
        test_size: Proportion of data for test set
        valid_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, valid_data, test_data)
    """
    print("\n[Step 2] Splitting data...")
    
    # First split: separate test set
    train_valid, test = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    valid_ratio_adjusted = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid, test_size=valid_ratio_adjusted, random_state=random_state
    )
    
    print(f"  Train: {len(train):,} (70%)")
    print(f"  Valid: {len(valid):,} (10%)")
    print(f"  Test:  {len(test):,} (20%)")
    
    return train, valid, test


def create_surprise_dataset(df: pd.DataFrame) -> Dataset:
    """
    Create Surprise library dataset object
    
    Args:
        df: Rating dataframe with columns (userId, movieId, rating)
        
    Returns:
        Surprise Dataset object ready for model training
    """
    reader = Reader(rating_scale=(config.MIN_RATING, config.MAX_RATING))
    return Dataset.load_from_df(
        df[['userId', 'movieId', 'rating']], 
        reader
    )


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate evaluation metrics: RMSE and MAE
    
    Args:
        predictions: Predicted ratings from the model
        actuals: Actual ratings from the dataset
        
    Returns:
        Tuple of (RMSE, MAE)
    """
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return rmse, mae


# ============================================================================
# Base Recommender Class
# ============================================================================

class BaseRecommender:
    """Base class for all recommender models"""
    
    def __init__(self):
        self.global_mean = 0.0  # Global average rating
        self.is_fitted = False  # Training status flag
        self.model = None       # Trained model object
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for test data
        
        Note: This method must be implemented by subclasses
        """
        raise NotImplementedError
    
    def clip_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Clip predictions to valid rating range
        
        Args:
            predictions: Raw prediction values
            
        Returns:
            Clipped predictions within [MIN_RATING, MAX_RATING]
        """
        return np.clip(predictions, config.MIN_RATING, config.MAX_RATING)


# ============================================================================
# Surprise Model Wrapper
# ============================================================================

class SurpriseWrapper(BaseRecommender):
    """Wrapper class for Surprise library models"""
    
    def __init__(self, trained_model, global_mean):
        """
        Initialize wrapper with trained Surprise model
        
        Args:
            trained_model: Trained Surprise model object
            global_mean: Global average rating for fallback predictions
        """
        super().__init__()
        self.model = trained_model
        self.global_mean = global_mean
        self.is_fitted = True
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for test data
        
        Args:
            test_df: Test data with columns (userId, movieId, rating)
            
        Returns:
            Array of predicted ratings
        """
        predictions = []
        
        for _, row in test_df.iterrows():
            try:
                # Use Surprise model for prediction
                pred = self.model.predict(row['userId'], row['movieId'])
                predictions.append(pred.est)
            except (PredictionImpossible, ValueError):
                # Fallback to global mean for cold-start cases
                predictions.append(self.global_mean)
        
        return self.clip_predictions(np.array(predictions))
    
    def recommend_for_user(self, user_id, all_items: List, rated_items: set, 
                          n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate movie recommendations for a specific user
        
        Args:
            user_id: User ID to generate recommendations for
            all_items: List of all available movie IDs
            rated_items: Set of movie IDs already rated by the user
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (movie_id, predicted_rating) tuples, sorted by rating
        """
        predictions = []
        
        # Predict ratings for all unrated movies
        for item_id in all_items:
            if item_id not in rated_items:
                try:
                    pred = self.model.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except (PredictionImpossible, ValueError):
                    predictions.append((item_id, self.global_mean))
        
        # Sort by predicted rating (descending) and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


# ============================================================================
# ALS Model Implementation
# ============================================================================

class ALSModel(BaseRecommender):
    """Alternating Least Squares (ALS) model for collaborative filtering"""
    
    def __init__(
        self, 
        factors: int = 50, 
        iterations: int = 15, 
        regularization: float = 0.01
    ):
        """
        Initialize ALS model
        
        Args:
            factors: Number of latent factors
            iterations: Number of training iterations
            regularization: L2 regularization parameter
        """
        super().__init__()
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=config.RANDOM_STATE,
            use_gpu=False
        )
    
    def fit(self, train_df: pd.DataFrame) -> 'ALSModel':
        """
        Train ALS model on rating data
        
        Args:
            train_df: Training data with columns (userId, movieId, rating)
            
        Returns:
            Self (trained model)
        """
        self.global_mean = train_df['rating'].mean()
        
        # Create mappings between IDs and indices
        users = sorted(train_df['userId'].unique())
        items = sorted(train_df['movieId'].unique())
        
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: i for i, it in enumerate(items)}
        
        # Build sparse user-item matrix
        user_indices = train_df['userId'].map(self.user_map).values
        item_indices = train_df['movieId'].map(self.item_map).values
        ratings = train_df['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(users), len(items))
        )
        
        # Train the model
        self.model.fit(self.user_item_matrix)
        self.is_fitted = True
        
        return self
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for test data
        
        Args:
            test_df: Test data with columns (userId, movieId, rating)
            
        Returns:
            Array of predicted ratings
        """
        predictions = []
        
        for _, row in test_df.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            
            # Handle cold-start problem (unseen users or items)
            if user_id not in self.user_map or item_id not in self.item_map:
                predictions.append(self.global_mean)
                continue
            
            # Compute prediction using dot product of latent factors
            user_idx = self.user_map[user_id]
            item_idx = self.item_map[item_id]
            
            user_vector = self.model.user_factors[user_idx]
            item_vector = self.model.item_factors[item_idx]
            
            pred = np.dot(user_vector, item_vector)
            
            # Scale prediction to rating range
            pred_scaled = pred * (config.MAX_RATING - config.MIN_RATING) / 10 + 3.0
            predictions.append(pred_scaled)
        
        return self.clip_predictions(np.array(predictions))
    
    def recommend_for_user(self, user_id, all_items: List, rated_items: set, 
                          n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate movie recommendations for a specific user
        
        Args:
            user_id: User ID to generate recommendations for
            all_items: List of all available movie IDs
            rated_items: Set of movie IDs already rated by the user
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (movie_id, predicted_rating) tuples, sorted by rating
        """
        # Handle cold-start case
        if user_id not in self.user_map:
            print(f"  Warning: User {user_id} not in training data. Using global mean.")
            unrated_items = [item for item in all_items if item not in rated_items]
            return [(item, self.global_mean) for item in unrated_items[:n_recommendations]]
        
        user_idx = self.user_map[user_id]
        user_vector = self.model.user_factors[user_idx]
        
        predictions = []
        for item_id in all_items:
            if item_id not in rated_items:
                if item_id not in self.item_map:
                    predictions.append((item_id, self.global_mean))
                else:
                    item_idx = self.item_map[item_id]
                    item_vector = self.model.item_factors[item_idx]
                    pred = np.dot(user_vector, item_vector)
                    pred_scaled = pred * (config.MAX_RATING - config.MIN_RATING) / 10 + 3.0
                    pred_clipped = np.clip(pred_scaled, config.MIN_RATING, config.MAX_RATING)
                    predictions.append((item_id, pred_clipped))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


# ============================================================================
# Hyperparameter Search Functions
# ============================================================================

def search_surprise_hyperparams(
    model_class: type,
    param_grid: Dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str
) -> Tuple[Any, Dict, float, float]:
    """
    Perform grid search for optimal hyperparameters (Surprise models)
    
    Args:
        model_class: Surprise model class (e.g., KNNBasic, SVD)
        param_grid: Dictionary of hyperparameter options to search
        train_df: Training data
        valid_df: Validation data for final evaluation
        model_name: Model name for logging
        
    Returns:
        Tuple of (best_model, best_params, valid_rmse, valid_mae)
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Hyperparameter Search")
    print(f"{'='*70}")
    
    # Calculate total number of parameter combinations
    n_combinations = 1
    for key, values in param_grid.items():
        if key != 'sim_options':
            n_combinations *= len(values)
        else:
            for sim_key, sim_values in values.items():
                if isinstance(sim_values, list):
                    n_combinations *= len(sim_values)
    
    print(f"  Parameter combinations to search: {n_combinations}")
    print(f"  Cross-validation: {config.CV_FOLDS}-Fold CV")
    print(f"  Total training runs: {n_combinations * config.CV_FOLDS}")
    print(f"  In progress...\n")
    
    data = create_surprise_dataset(train_df)
    start_time = time.time()
    
    try:
        # Execute grid search with cross-validation
        grid_search = GridSearchCV(
            model_class,
            param_grid,
            measures=['rmse'],
            cv=config.CV_FOLDS,
            n_jobs=config.N_JOBS,
            joblib_verbose=0
        )
        grid_search.fit(data)
        
        best_params = grid_search.best_params['rmse']
        best_rmse = grid_search.best_score['rmse']
        
    except TypeError:
        # Compatibility with older Surprise versions
        print("  (Old Surprise version detected)")
        
        grid_search = GridSearchCV(
            model_class,
            param_grid,
            cv=config.CV_FOLDS,
            n_jobs=config.N_JOBS
        )
        grid_search.fit(data)
        
        best_params = grid_search.best_params
        best_rmse = grid_search.best_score
    
    elapsed = time.time() - start_time
    
    print(f"\n  Completion time: {elapsed:.1f}s")
    print(f"  Best CV RMSE: {best_rmse:.4f}")
    print(f"  Best parameters: {best_params}")
    
    # Retrain model with best parameters on full training data
    print("  Retraining on full data...", end=" ", flush=True)
    trainset = data.build_full_trainset()
    best_model = model_class(**best_params)
    best_model.fit(trainset)
    print("Done")
    
    # Evaluate on validation set
    print("  Evaluating on validation data...", end=" ", flush=True)
    global_mean = train_df['rating'].mean()
    model = SurpriseWrapper(best_model, global_mean)
    valid_preds = model.predict(valid_df)
    valid_rmse, valid_mae = calculate_metrics(valid_preds, valid_df['rating'].values)
    print(f"Done (RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f})")
    
    return best_model, best_params, valid_rmse, valid_mae


def search_als_hyperparams(
    param_grid: List[Dict],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_name: str
) -> Tuple[Dict, float, float]:
    """
    Perform grid search for optimal hyperparameters (ALS model)
    
    Args:
        param_grid: List of parameter dictionaries to search
        train_df: Training data
        valid_df: Validation data for evaluation
        model_name: Model name for logging
        
    Returns:
        Tuple of (best_params, valid_rmse, valid_mae)
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Hyperparameter Search")
    print(f"{'='*70}")
    
    print(f"  Parameter combinations to search: {len(param_grid)}")
    
    best_score = float('inf')
    best_params = None
    best_mae = None
    
    # Test each parameter combination
    for idx, params in enumerate(param_grid, 1):
        print(f"  [{idx}/{len(param_grid)}] {params}", end=" ", flush=True)
        
        try:
            model = ALSModel(**params)
            model.fit(train_df)
            preds = model.predict(valid_df)
            rmse, mae = calculate_metrics(preds, valid_df['rating'].values)
            
            if rmse < best_score:
                best_score = rmse
                best_mae = mae
                best_params = params
            
            print(f"→ RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        except Exception as e:
            print(f"→ Error: {str(e)}")
    
    print(f"\n  Best parameters: {best_params}")
    print(f"  Validation RMSE: {best_score:.4f}, MAE: {best_mae:.4f}")
    
    return best_params, best_score, best_mae


# ============================================================================
# Model Evaluation Functions
# ============================================================================

def evaluate_surprise_model(
    best_model: Any,
    train_full: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    valid_rmse: float,
    valid_mae: float
) -> Dict:
    """
    Evaluate Surprise model on test set
    
    Args:
        best_model: Trained model with optimal hyperparameters
        train_full: Combined train + validation data
        valid_df: Validation data (for reference)
        test_df: Test data for final evaluation
        model_name: Model name for logging
        valid_rmse: Validation RMSE
        valid_mae: Validation MAE
        
    Returns:
        Dictionary containing evaluation results and trained model
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Test Data Evaluation")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Retrain on full training data (train + validation)
    print("  Retraining on full data...", end=" ", flush=True)
    data_full = create_surprise_dataset(train_full)
    trainset_full = data_full.build_full_trainset()
    best_model.fit(trainset_full)
    retrain_time = time.time() - start_time
    print(f"Done ({retrain_time:.1f}s)")
    
    # Generate predictions for test set
    global_mean = train_full['rating'].mean()
    model = SurpriseWrapper(best_model, global_mean)
    
    print("  Predicting on test data...", end=" ", flush=True)
    pred_start = time.time()
    test_preds = model.predict(test_df)
    pred_time = time.time() - pred_start
    print(f"Done ({pred_time:.1f}s)")
    
    # Calculate test metrics
    test_rmse, test_mae = calculate_metrics(test_preds, test_df['rating'].values)
    total_time = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"     Valid RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f}")
    print(f"     Test RMSE:  {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"  Time taken: {total_time:.1f}s")
    
    return {
        'model': model_name,
        'valid_rmse': valid_rmse,
        'valid_mae': valid_mae,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'time_sec': total_time,
        'trained_model': model  # Store for recommendation generation
    }


def evaluate_als_model(
    best_params: Dict,
    train_full: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    valid_rmse: float,
    valid_mae: float
) -> Dict:
    """
    Evaluate ALS model on test set
    
    Args:
        best_params: Optimal hyperparameters
        train_full: Combined train + validation data
        valid_df: Validation data (for reference)
        test_df: Test data for final evaluation
        model_name: Model name for logging
        valid_rmse: Validation RMSE
        valid_mae: Validation MAE
        
    Returns:
        Dictionary containing evaluation results and trained model
    """
    print(f"\n{'='*70}")
    print(f"[{model_name}] Test Data Evaluation")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Train model with best parameters on full training data
    print("  Training...", end=" ", flush=True)
    model = ALSModel(**best_params)
    model.fit(train_full)
    train_time = time.time() - start_time
    print(f"Done ({train_time:.1f}s)")
    
    # Generate predictions for test set
    print("  Predicting...", end=" ", flush=True)
    pred_start = time.time()
    test_preds = model.predict(test_df)
    pred_time = time.time() - pred_start
    print(f"Done ({pred_time:.1f}s)")
    
    # Calculate test metrics
    test_rmse, test_mae = calculate_metrics(test_preds, test_df['rating'].values)
    total_time = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"     Valid RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f}")
    print(f"     Test RMSE:  {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"  Time taken: {total_time:.1f}s")
    
    return {
        'model': model_name,
        'valid_rmse': valid_rmse,
        'valid_mae': valid_mae,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'time_sec': total_time,
        'trained_model': model  # Store for recommendation generation
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_results(results_df: pd.DataFrame, output_path: str = 'cf_results_visualization.png'):
    """
    Visualize experiment results using matplotlib
    
    Creates comprehensive visualization including:
    - Test RMSE comparison
    - Test MAE comparison
    - Validation vs Test RMSE comparison
    - Execution time comparison
    
    Args:
        results_df: DataFrame containing evaluation results
        output_path: Path to save the visualization image
    """
    print("\n" + "="*70)
    print("Visualizing Results")
    print("="*70)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MovieLens Collaborative Filtering - Test Results Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color palette for consistent visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    # Plot 1: Test RMSE comparison (horizontal bar chart)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(results_df['model'], results_df['test_rmse'], color=colors)
    ax1.set_xlabel('Test RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Test RMSE by Model (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels to bars
    for i, (bar, val) in enumerate(zip(bars1, results_df['test_rmse'])):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Test MAE comparison (horizontal bar chart)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(results_df['model'], results_df['test_mae'], color=colors)
    ax2.set_xlabel('Test MAE', fontsize=12, fontweight='bold')
    ax2.set_title('Test MAE by Model (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels to bars
    for i, (bar, val) in enumerate(zip(bars2, results_df['test_mae'])):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Validation vs Test RMSE comparison (grouped bar chart)
    ax3 = axes[1, 0]
    x = np.arange(len(results_df))
    width = 0.35
    
    bars3_1 = ax3.bar(x - width/2, results_df['valid_rmse'], width, 
                      label='Validation RMSE', alpha=0.8, color='skyblue')
    bars3_2 = ax3.bar(x + width/2, results_df['test_rmse'], width, 
                      label='Test RMSE', alpha=0.8, color='coral')
    
    ax3.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax3.set_title('Validation vs Test RMSE Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Execution time comparison (horizontal bar chart)
    ax4 = axes[1, 1]
    bars4 = ax4.barh(results_df['model'], results_df['time_min'], color=colors)
    ax4.set_xlabel('Execution Time (minutes)', fontsize=12, fontweight='bold')
    ax4.set_title('Training & Evaluation Time by Model', fontsize=13, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels to bars
    for i, (bar, val) in enumerate(zip(bars4, results_df['time_min'])):
        ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}m', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved: {output_path}")
    
    # Additional visualization: Performance map (RMSE vs MAE scatter plot)
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(results_df['test_rmse'], results_df['test_mae'], 
                        s=500, c=range(len(results_df)), cmap='viridis', 
                        alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add model name labels
    for idx, row in results_df.iterrows():
        ax.annotate(row['model'], 
                   (row['test_rmse'], row['test_mae']),
                   fontsize=11, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Test RMSE', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test MAE', fontsize=13, fontweight='bold')
    ax.set_title('Test RMSE vs MAE - Model Performance Map', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight optimal region (lower-left corner)
    ax.axvline(results_df['test_rmse'].min(), color='red', linestyle='--', 
              alpha=0.3, label='Best RMSE')
    ax.axhline(results_df['test_mae'].min(), color='blue', linestyle='--', 
              alpha=0.3, label='Best MAE')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cf_results_performance_map.png', dpi=300, bbox_inches='tight')
    print(f"  Performance map saved: cf_results_performance_map.png")
    
    print("  Visualization completed!")


# ============================================================================
# Recommendation Function
# ============================================================================

def recommend_movies_for_user(
    best_model_info: Dict,
    user_id: int,
    train_full: pd.DataFrame,
    pivot_df: pd.DataFrame,
    n_recommendations: int = 10
):
    """
    Generate movie recommendations for a specific user using the best model
    
    Args:
        best_model_info: Dictionary containing best model information
        user_id: Target user ID for recommendations
        train_full: Full training dataset
        pivot_df: Original pivot table (user x movie)
        n_recommendations: Number of movies to recommend
    """
    print("\n" + "="*70)
    print(f"Movie Recommendations for User {user_id}")
    print("="*70)
    print(f"Using Best Model: {best_model_info['model']}")
    print(f"  Test RMSE: {best_model_info['test_rmse']:.4f}")
    print(f"  Test MAE: {best_model_info['test_mae']:.4f}")
    print("-"*70)
    
    # Get trained model
    model = best_model_info['trained_model']
    
    # Get all available movies
    all_movies = list(pivot_df.columns)
    
    # Get movies already rated by the user
    user_ratings = train_full[train_full['userId'] == user_id]
    rated_movies = set(user_ratings['movieId'].values)
    
    print(f"\nUser {user_id} Statistics:")
    print(f"  Total movies rated: {len(rated_movies)}")
    if len(user_ratings) > 0:
        print(f"  Average rating: {user_ratings['rating'].mean():.2f}")
        print(f"  Rating range: {user_ratings['rating'].min():.1f} - {user_ratings['rating'].max():.1f}")
    
    # Generate recommendations
    print(f"\nGenerating Top {n_recommendations} recommendations...")
    recommendations = model.recommend_for_user(user_id, all_movies, rated_movies, n_recommendations)
    
    # Display recommendations
    print(f"\n{'='*70}")
    print(f"Top {n_recommendations} Movie Recommendations for User {user_id}:")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Movie ID':<15} {'Predicted Rating':<20}")
    print("-"*70)
    
    for rank, (movie_id, pred_rating) in enumerate(recommendations, 1):
        print(f"{rank:<6} {movie_id:<15} {pred_rating:.4f}")
    
    # Save recommendations to CSV
    rec_df = pd.DataFrame(recommendations, columns=['movieId', 'predicted_rating'])
    rec_df['rank'] = range(1, len(rec_df) + 1)
    rec_df['userId'] = user_id
    rec_df = rec_df[['rank', 'userId', 'movieId', 'predicted_rating']]
    
    output_file = f'recommendations_user_{user_id}.csv'
    rec_df.to_csv(output_file, index=False)
    print(f"\nRecommendations saved: {output_file}")
    
    return rec_df


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def run_experiment(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete experimental pipeline
    
    Pipeline stages:
    1. Data preparation and splitting
    2. Hyperparameter search for all models
    3. Model evaluation on test set
    4. Result visualization
    5. Top-10 recommendations using best model
    
    Args:
        pivot_df: User x Movie pivot table with ratings
        
    Returns:
        DataFrame containing evaluation results for all models
    """
    experiment_start = time.time()
    
    # Stage 1: Prepare data
    ratings_df = transform_data(pivot_df)
    train_df, valid_df, test_df = split_data(ratings_df)
    train_full = pd.concat([train_df, valid_df], ignore_index=True)
    
    # Stage 2: Configure models based on mode
    if config.FAST_MODE:
        print("\n" + "="*70)
        print("FAST MODE (Limited Parameter Search)")
        print("="*70)
        
        model_list = [
            {
                'name': 'Item-Item kNN',
                'class': KNNBasic,
                'param_grid': {
                    'k': [30, 50],
                    'sim_options': {
                        'name': ['pearson'],
                        'user_based': [False],
                        'min_support': [3],
                        'shrinkage': [100]
                    }
                },
                'type': 'surprise'
            },
            {
                'name': 'User-User kNN',
                'class': KNNBasic,
                'param_grid': {
                    'k': [30, 50],
                    'sim_options': {
                        'name': ['pearson'],
                        'user_based': [True],
                        'min_support': [3],
                        'shrinkage': [100]
                    }
                },
                'type': 'surprise'
            },
            {
                'name': 'Slope-One',
                'class': SlopeOne,
                'param_grid': {},
                'type': 'surprise'
            },
            {
                'name': 'Bias-SVD',
                'class': SVD,
                'param_grid': {
                    'n_factors': [50, 64],
                    'n_epochs': [20, 25],
                    'lr_all': [0.005],
                    'reg_all': [0.02]
                },
                'type': 'surprise'
            },
            {
                'name': 'NMF',
                'class': SurpriseNMF,
                'param_grid': {
                    'n_factors': [40, 50],
                    'n_epochs': [150, 200],
                    'reg_pu': [0.06],
                    'reg_qi': [0.06]
                },
                'type': 'surprise'
            },
            {
                'name': 'ALS',
                'class': ALSModel,
                'param_grid': [
                    {'factors': 40, 'iterations': 10, 'regularization': 0.01},
                    {'factors': 50, 'iterations': 15, 'regularization': 0.01}
                ],
                'type': 'als'
            }
        ]
    else:
        print("\n" + "="*70)
        print("FULL MODE (Extensive Parameter Search)")
        print("="*70)
        
        model_list = [
            {
                'name': 'Item-Item kNN',
                'class': KNNBasic,
                'param_grid': {
                    'k': [20, 30, 50, 70, 100],
                    'sim_options': {
                        'name': ['pearson', 'cosine'],
                        'user_based': [False],
                        'min_support': [1, 2, 3, 5],
                        'shrinkage': [30, 50, 100, 150]
                    }
                },
                'type': 'surprise'
            },
            {
                'name': 'User-User kNN',
                'class': KNNBasic,
                'param_grid': {
                    'k': [20, 30, 50, 70, 100],
                    'sim_options': {
                        'name': ['pearson', 'cosine'],
                        'user_based': [True],
                        'min_support': [1, 2, 3, 5],
                        'shrinkage': [30, 50, 100, 150]
                    }
                },
                'type': 'surprise'
            },
            {
                'name': 'Slope-One',
                'class': SlopeOne,
                'param_grid': {},
                'type': 'surprise'
            },
            {
                'name': 'Bias-SVD',
                'class': SVD,
                'param_grid': {
                    'n_factors': [20, 30, 50, 70, 100],
                    'n_epochs': [10, 15, 20, 25, 30],
                    'lr_all': [0.001, 0.002, 0.005, 0.007, 0.01],
                    'reg_all': [0.01, 0.015, 0.02, 0.025, 0.04]
                },
                'type': 'surprise'
            },
            {
                'name': 'NMF',
                'class': SurpriseNMF,
                'param_grid': {
                    'n_factors': [20, 30, 50, 70, 100],
                    'n_epochs': [100, 150, 180, 200, 250],
                    'reg_pu': [0.04, 0.06, 0.07, 0.08, 0.1],
                    'reg_qi': [0.04, 0.06, 0.07, 0.08, 0.1]
                },
                'type': 'surprise'
            },
            {
                'name': 'ALS',
                'class': ALSModel,
                'param_grid': [
                    {'factors': f, 'iterations': i, 'regularization': r}
                    for f in [20, 30, 50, 70, 100]
                    for i in [8, 10, 15, 18, 20]
                    for r in [0.005, 0.01, 0.012, 0.015, 0.02]
                ],
                'type': 'als'
            }
        ]
    
    # Stage 3: Train and evaluate all models
    all_results = []
    
    print("\n" + "="*70)
    print("Starting Evaluation of All Models")
    print("="*70)
    
    for idx, model_info in enumerate(model_list, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Progress: {idx}/{len(model_list)} - {model_info['name']}")
        print(f"{'#'*70}")
        
        if model_info['type'] == 'surprise':
            # Surprise library models
            best_model, best_params, valid_rmse, valid_mae = search_surprise_hyperparams(
                model_info['class'],
                model_info['param_grid'],
                train_df,
                valid_df,
                model_info['name']
            )
            
            result = evaluate_surprise_model(
                best_model,
                train_full,
                valid_df,
                test_df,
                model_info['name'],
                valid_rmse,
                valid_mae
            )
        
        else:  # ALS model
            best_params, valid_rmse, valid_mae = search_als_hyperparams(
                model_info['param_grid'],
                train_df,
                valid_df,
                model_info['name']
            )
            
            result = evaluate_als_model(
                best_params,
                train_full,
                valid_df,
                test_df,
                model_info['name'],
                valid_rmse,
                valid_mae
            )
        
        all_results.append(result)
    
    total_time = time.time() - experiment_start
    
    # Stage 4: Display and save results
    print("\n\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results).sort_values('test_rmse')
    results_df['time_min'] = results_df['time_sec'] / 60
    
    print("\nOverall Performance (Sorted by Test RMSE):")
    display_df = results_df[['model', 'valid_rmse', 'valid_mae', 'test_rmse', 'test_mae', 'time_min']].copy()
    print(display_df.to_string(index=False))
    
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    
    # Save results to CSV
    display_df.to_csv('cf_results.csv', index=False)
    print("\nResults saved: cf_results.csv")
    
    # Display ranking
    print("\n" + "="*80)
    print("Performance Ranking")
    print("="*80)
    
    print("\n[Ranked by Test RMSE]")
    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"  {rank}. {row['model']:20s} "
              f"Valid: {row['valid_rmse']:.4f}/{row['valid_mae']:.4f}, "
              f"Test: {row['test_rmse']:.4f}/{row['test_mae']:.4f}")
    
    # Stage 5: Visualize results
    visualize_results(display_df)
    
    # Stage 6: Generate recommendations using best model
    best_model_info = all_results[results_df.index[0]]
    
    # Select first user as sample
    sample_user = train_full['userId'].iloc[0]
    recommend_movies_for_user(best_model_info, sample_user, train_full, pivot_df, n_recommendations=10)
    
    return results_df


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MovieLens Collaborative Filtering Recommender System")
    print("="*80)
    print(f"Configuration: FAST_MODE={config.FAST_MODE}")
    print(f"Cross-validation: {config.CV_FOLDS}-Fold CV")
    print(f"Parallel processing: Using all CPU cores")
    print(f"Evaluation metrics: RMSE, MAE")
    print("="*80)
    
    print("\nLoading data...")
    pivot_df = pd.read_csv('MovieLens_CF_preprocessed.csv', index_col=0)
    print(f"Data size: {pivot_df.shape}")
    print(f"  Users: {len(pivot_df.index):,}")
    print(f"  Movies: {len(pivot_df.columns):,}")
    
    # Run complete experiment pipeline
    results = run_experiment(pivot_df)
    
    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  - cf_results.csv: Performance metrics table")
    print("  - cf_results_visualization.png: Results visualization")
    print("  - cf_results_performance_map.png: Performance comparison map")
    print("  - recommendations_user_X.csv: Movie recommendations for sample user")