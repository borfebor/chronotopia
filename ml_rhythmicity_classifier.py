"""
ML Rhythmicity Classifier
Corrected version with proper feature name handling
"""

import joblib
import pickle
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical_detector import HierarchicalRhythmicityDetector


class MLRhythmicityClassifier:
    """
    Wrapper for trained ML classifier for rhythmicity detection.
    
    Usage:
        classifier = MLRhythmicityClassifier('rhythmicity_classifier.pkl')
        result = classifier.predict(signal, time)
        print(result['is_rhythmic'], result['probability_rhythmic'])
    """
    
    def __init__(self, model_path='rhythmicity_classifier.pkl', 
                 feature_names_path='feature_names.pkl',
                 scaler_path='feature_scaler.pkl',
                 metadata_path='model_metadata.pkl'):
        """
        Load trained model and associated files.
        
        Args:
            model_path: Path to saved model (.pkl)
            feature_names_path: Path to saved feature names (.pkl)
            scaler_path: Path to scaler (optional, only if model uses scaling)
            metadata_path: Path to model metadata (optional)
        """
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Load feature names (REQUIRED)
        try:
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"Loaded {len(self.feature_names)} feature names")
        except FileNotFoundError:
            print(f"WARNING: {feature_names_path} not found!")
            print("Creating feature names from model training...")
            # Emergency fallback - get from first prediction
            self.feature_names = None
        
        # Load scaler (optional)
        try:
            self.scaler = joblib.load(scaler_path)
            print("Loaded feature scaler")
            self.uses_scaler = True
        except FileNotFoundError:
            self.scaler = None
            self.uses_scaler = False
        
        # Load metadata (optional)
        try:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
            print(f"Training accuracy: {self.metadata.get('accuracy', 0):.3f}")
        except FileNotFoundError:
            self.metadata = {}
    
    def predict(self, signal, time, period_range=(20, 28)):
        """
        Predict if signal is rhythmic.
        
        Args:
            signal: Time series data (array-like)
            time: Time points (array-like)
            period_range: Period range to analyze (tuple)
        
        Returns:
            dict with:
                - is_rhythmic: bool
                - probability_rhythmic: float (0-1)
                - probability_arrhythmic: float (0-1)
                - confidence: str ('high', 'medium', 'low')
        """
        try:
            # Extract metrics using hierarchical detector
            detector = HierarchicalRhythmicityDetector(
                signal, 
                time, 
                period_range=period_range
            )
            metrics = detector.extract_all_metrics()
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame([metrics])
            
            # If feature names not loaded, use all metrics
            if self.feature_names is None:
                print("WARNING: Using all metrics as features (not recommended)")
                # Remove non-feature columns
                exclude_cols = ['sample_id', 'label', 'error']
                self.feature_names = [col for col in metrics_df.columns 
                                     if col not in exclude_cols]
            
            # Select only the features used in training
            # Handle missing features
            missing_features = set(self.feature_names) - set(metrics_df.columns)
            if missing_features:
                print(f"WARNING: Missing features: {missing_features}")
                # Add missing features with 0
                for feat in missing_features:
                    metrics_df[feat] = 0
            
            X = metrics_df[self.feature_names]
            
            # Handle NaN and inf values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)  # Replace NaN with 0 (or use median from training)
            
            # Scale if needed
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Determine confidence
            max_prob = max(probabilities)
            if max_prob > 0.8:
                confidence = 'high'
            elif max_prob > 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'is_rhythmic': bool(prediction),
                'probability_rhythmic': float(probabilities[1]),
                'probability_arrhythmic': float(probabilities[0]),
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"ERROR in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return safe default
            return {
                'is_rhythmic': False,
                'probability_rhythmic': 0.0,
                'probability_arrhythmic': 1.0,
                'confidence': 'error',
                'error': str(e)
            }
    
    def predict_batch(self, df, time_col, data_cols, period_range=(20, 28)):
        """
        Predict for multiple samples.
        
        Args:
            df: DataFrame with time and data columns
            time_col: Name of time column
            data_cols: List of data column names
            period_range: Period range to analyze
        
        Returns:
            DataFrame with predictions for each sample
        """
        results = []
        
        for col in data_cols:
            print(f"Predicting {col}...")
            
            result = self.predict(
                df[col].values,
                df[time_col].values,
                period_range=period_range
            )
            
            result['sample'] = col
            results.append(result)
        
        return pd.DataFrame(results)


def save_trained_model(model, X_train, scaler=None, model_name='rhythmicity_classifier'):
    """
    Helper function to save model with all necessary files.
    
    Args:
        model: Trained sklearn model
        X_train: Training features DataFrame (to get feature names)
        scaler: Optional fitted scaler
        model_name: Base name for saved files
    
    Saves:
        - {model_name}.pkl: The trained model
        - feature_names.pkl: List of feature names
        - feature_scaler.pkl: Scaler (if provided)
        - model_metadata.pkl: Model information
    """
    import joblib
    import pickle
    
    # Save model
    model_path = f'{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Saved {len(feature_names)} feature names to feature_names.pkl")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = 'feature_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'uses_scaler': scaler is not None
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to model_metadata.pkl")
    
    print("\nTo load this model:")
    print(f"classifier = MLRhythmicityClassifier('{model_path}')")


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Test the classifier
    try:
        classifier = MLRhythmicityClassifier('rhythmicity_classifier.pkl')
        
        # Generate test signal
        t = np.arange(0, 240, 0.5)
        signal = 2 * np.sin(2*np.pi*t/24) + np.random.normal(0, 0.3, len(t))
        
        # Predict
        result = classifier.predict(signal, t)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Rhythmic: {result['is_rhythmic']}")
        print(f"Probability (rhythmic): {result['probability_rhythmic']:.3f}")
        print(f"Probability (arrhythmic): {result['probability_arrhythmic']:.3f}")
        print(f"Confidence: {result['confidence']}")
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find model files: {e}")
        print("\nMake sure you have:")
        print("  - rhythmicity_classifier.pkl")
        print("  - feature_names.pkl")
        print("  - feature_scaler.pkl (if using scaled features)")
