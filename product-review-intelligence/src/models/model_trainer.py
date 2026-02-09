"""
Model training for sentiment classification
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..nlp.text_preprocessor import ReviewPreprocessor
from ..nlp.feature_extractor import ReviewFeatureExtractor


class SentimentModelTrainer:
    """Train and evaluate sentiment classification models"""
    
    def __init__(self, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_extractor = ReviewFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize model configurations"""
        
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', None]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Multinomial NB': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0]
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple:
        """Prepare data for training"""
        
        print("Preprocessing text data...")
        preprocessor = ReviewPreprocessor()
        processed_texts = preprocessor.batch_preprocess(texts)
        
        print("Extracting features...")
        features_dict = self.feature_extractor.extract_all_features(
            processed_texts,
            include_tfidf=True,
            include_topics=True,
            include_sentiment=True,
            include_readability=True
        )
        
        X = features_dict['combined']
        feature_names = features_dict['feature_names']
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train all models with hyperparameter tuning"""
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print('-'*30)
            
            try:
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['params'],
                    cv=self.cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store results
                results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_,
                    'training_time': grid_search.refit_time_
                }
                
                print(f"✓ Best F1 Score: {grid_search.best_score_:.4f}")
                print(f"✓ Best parameters: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
                results[name] = {
                    'model': None,
                    'error': str(e)
                }
        
        self.results = results
        return results
    
    def select_best_model(self) -> Tuple[Any, str]:
        """Select the best performing model"""
        
        valid_results = {k: v for k, v in self.results.items() 
                        if v.get('model') is not None}
        
        if not valid_results:
            raise ValueError("No successful model training results found")
        
        best_model_name = max(
            valid_results.items(), 
            key=lambda x: x[1]['best_score']
        )[0]
        
        self.best_model = valid_results[best_model_name]['model']
        
        print(f"\n{'='*60}")
        print("MODEL SELECTION RESULTS")
        print('='*60)
        
        for name, result in valid_results.items():
            score = result['best_score']
            mark = "★" if name == best_model_name else " "
            print(f"{mark} {name:20s} | CV F1 Score: {score:.4f}")
        
        print(f"\nSelected Best Model: {best_model_name}")
        print(f"Validation F1 Score: {valid_results[best_model_name]['best_score']:.4f}")
        print('='*60)
        
        return self.best_model, best_model_name
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate best model on test set"""
        
        if self.best_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
            precision_recall_curve, average_precision_score
        )
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=self.label_encoder.classes_,
                                                          output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            metrics['precision_recall_curve'] = precision_recall_curve(y_test, y_pred_proba)
        
        print("\n" + "="*60)
        print("TEST SET PERFORMANCE")
        print("="*60)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"AUC-ROC:     {metrics['roc_auc']:.4f}")
        if 'average_precision' in metrics:
            print(f"Avg Precision: {metrics['average_precision']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        return metrics
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the best model"""
        
        if self.best_model is None:
            raise ValueError("No model selected.")
        
        importance_df = pd.DataFrame()
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
        # For linear models, use coefficients
        elif hasattr(self.best_model, 'coef_'):
            coef = self.best_model.coef_[0]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath: str, metadata: Dict = None):
        """Save the trained model and all components"""
        
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'model': self.best_model,
            'feature_extractor': self.feature_extractor,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'results': self.results,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.results = model_data.get('results', {})
        
        print(f"✓ Model loaded from {filepath}")
        
        return self.best_model
    
    def predict_sentiment(self, texts: List[str]) -> Tuple:
        """Predict sentiment for new texts"""
        
        if self.best_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Preprocess
        preprocessor = ReviewPreprocessor()
        processed_texts = preprocessor.batch_preprocess(texts)
        
        # Extract features
        features_dict = self.feature_extractor.extract_all_features(processed_texts)
        X = features_dict['combined']
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled) if hasattr(self.best_model, 'predict_proba') else None
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'processed_text': processed_texts[i],
                'sentiment': predicted_labels[i]
            }
            
            if probabilities is not None:
                result['confidence'] = probabilities[i].max()
                result['probabilities'] = dict(zip(
                    self.label_encoder.classes_, 
                    probabilities[i]
                ))
            
            results.append(result)
        
        return results