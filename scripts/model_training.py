import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
import sys
import pandas as pd
import numpy as np
import configparser
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
import tensorflow as tf
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
load_model=tf.keras.models.load_model
Tokenizer=tf.keras.preprocessing.text.Tokenizer
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
Sequential=tf.keras.models.Sequential
#from tensorflow.keras.models import Sequential
Embedding=tf.keras.layers.Embedding
LSTM=tf.keras.layers.LSTM
Dense=tf.keras.layers.Dense
Dropout =tf.keras.layers.Dropout
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 # Suppress INFO and WARNING logs
from keras.callbacks import ModelCheckpoint
# Add the project root to the path
import mlflow.keras
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read(os.path.join(project_root, "config/config.ini"))

logger = get_logger(__name__)

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    logger.info(f"Loading data from: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if needed columns exist
        if 'headline' not in df.columns or 'content' not in df.columns:
            # Try different column names based on potential structure
            if 'title' in df.columns:
                df.rename(columns={'title': 'headline'}, inplace=True)
            
            if 'article_content' in df.columns:
                df.rename(columns={'article_content': 'content'}, inplace=True)
        
        # Ensure we have a sentiment column
        sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
        if not sentiment_columns:
            logger.error("No sentiment column found in the dataset")
            return None
            
        sentiment_col = sentiment_columns[0]
        logger.info(f"Using {sentiment_col} as the sentiment column")
        
        # Create a clean dataframe with needed columns
        cleaned_df = pd.DataFrame({
            'text': df['content'].fillna('').astype(str),
            'headline': df['headline'].fillna('').astype(str),
            'sentiment': df[sentiment_col]
        })
        
        # Convert sentiment labels to numerical values if they're categorical
        sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        
        if cleaned_df['sentiment'].dtype == 'object':
            cleaned_df['sentiment'] = cleaned_df['sentiment'].map(sentiment_mapping)
            logger.info("Converted categorical sentiment labels to numerical values")
        
        # Drop rows with missing sentiment
        cleaned_df = cleaned_df.dropna(subset=['sentiment'])
        
        # Ensure sentiment values are integers
        cleaned_df['sentiment'] = cleaned_df['sentiment'].astype(int)
        
        logger.info(f"Preprocessed data shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

def tokenize_text(texts, max_features=10000, max_len=200):
    """Tokenize and pad the text data"""
    logger.info(f"Tokenizing text with max_features={max_features}, max_len={max_len}")
    
    try:
        tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        logger.info(f"Created padded sequences with shape: {padded_sequences.shape}")
        return padded_sequences, tokenizer
        
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return None, None

def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=64, max_len=200):
    """Build the LSTM model for sentiment analysis"""
    logger.info(f"Building LSTM model with vocab_size={vocab_size}")
    
    try:
        # Try to use GPU if available
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        logger.info(f"Using device: {device}")
        
        with tf.device(device):
            model = Sequential([
                Embedding(vocab_size, embedding_dim, input_length=max_len),
                LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                LSTM(lstm_units, dropout=0.2),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(3, activation='softmax')  # 3 classes: Negative, Neutral, Positive
            ])
            
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        
        logger.info("LSTM model built successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error building model: {e}")
        return None

def evaluate_model(model, X, y, label):
    """Evaluate the model and return metrics"""
    logger.info(f"Evaluating model on {label} data")
    
    try:
        # Use batches for prediction
        batch_size = int(config["MODEL"]["batch_size"])
        y_pred_proba = model.predict(X, batch_size=batch_size)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        logger.info(f"{label} Accuracy: {accuracy:.4f}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {label}')
        plt.tight_layout()
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(project_root, "models/plots")
        create_directory_if_not_exists(plots_dir)
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'confusion_matrix_{label.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return accuracy, report, plot_path
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None, None, None

def train_and_evaluate():
    """Train and evaluate the sentiment analysis model with MLflow tracking"""
    logger.info("Starting model training")
    
    # Setup MLflow
    mlflow_uri = config["MLFLOW"]["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create the experiment if it doesn't exist
    experiment_name = config["MLFLOW"]["experiment_name"]
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Configure model parameters
    max_features = int(config["MODEL"]["max_features"])
    max_len = int(config["MODEL"]["max_len"])
    embedding_dim = int(config["MODEL"]["embedding_dim"])
    lstm_units = int(config["MODEL"]["lstm_units"])
    batch_size = int(config["MODEL"]["batch_size"])
    epochs = int(config["MODEL"]["epochs"])
    test_size = float(config["MODEL"]["test_size"])
    
    # Create model directory if it doesn't exist
    model_dir = config["MODEL"]["model_dir"]
    create_directory_if_not_exists(model_dir)
    
    # Load training data
    data_file = os.path.join(project_root, "data/processed/training_data.csv")
    df = load_and_preprocess_data(data_file)
    
    if df is None:
        logger.error("Failed to load and preprocess data")
        return False
    
    with mlflow.start_run(run_name="news_sentiment_model"):
        # Log parameters
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        
        # Prepare training data
        X_text = df['text'].values
        y = df['sentiment'].values
        
        # Tokenize text
        X_seq, tokenizer = tokenize_text(X_text, max_features, max_len)
        if X_seq is None or tokenizer is None:
            logger.error("Failed to tokenize text")
            return False
        
        # Save tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Saved tokenizer to {tokenizer_path}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Convert to TensorFlow datasets for optimized loading
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Build model
        model = build_lstm_model(max_features+1, embedding_dim, lstm_units, max_len)
        if model is None:
            logger.error("Failed to build model")
            return False
        
        # Set up callbacks
        checkpoint_path = config["MODEL"]["checkpoint_path"]
        
        print(f"Checkpoint path: {checkpoint_path}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path, 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
                ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting model training")
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        
        # Evaluate model
        train_accuracy, train_report, train_plot = evaluate_model(model, X_train, y_train, "Training")
        val_accuracy, val_report, val_plot = evaluate_model(model, X_val, y_val, "Validation")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        
        # Log detailed metrics from reports
        for label in ['Negative', 'Neutral', 'Positive']:
            for metric in ['precision', 'recall', 'f1-score']:
                mlflow.log_metric(f"train_{label.lower()}_{metric}", train_report[label][metric])
                mlflow.log_metric(f"val_{label.lower()}_{metric}", val_report[label][metric])
        
        # Log confusion matrix plots
        if train_plot:
            mlflow.log_artifact(train_plot)
        if val_plot:
            mlflow.log_artifact(val_plot)
        
        # Log model
        model_path = os.path.join(model_dir, "sentiment_model.keras")
        model.save(model_path)
        mlflow.log_artifact(model_path)

        
        # Log training history
        for epoch, (loss, val_loss, accuracy, val_accuracy) in enumerate(zip(
            history.history['loss'], 
            history.history['val_loss'],
            history.history['accuracy'], 
            history.history['val_accuracy']
        )):
            mlflow.log_metrics({
                "epoch_loss": loss,
                "epoch_val_loss": val_loss,
                "epoch_accuracy": accuracy,
                "epoch_val_accuracy": val_accuracy
            }, step=epoch)
        
        # Plot and log training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plots_dir = os.path.join(project_root, "models/plots")
        create_directory_if_not_exists(plots_dir)
        history_plot_path = os.path.join(plots_dir, 'training_history.png')
        plt.savefig(history_plot_path)
        plt.close()
        
        mlflow.log_artifact(history_plot_path)
        
        logger.info("Model training and evaluation completed")
        return True

if __name__ == "__main__":
    logger.info("Starting sentiment analysis model training pipeline")
    success = train_and_evaluate()
    
    if success:
        logger.info("Model training pipeline completed successfully")
    else:
        logger.error("Model training pipeline failed")