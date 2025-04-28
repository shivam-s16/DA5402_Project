import argparse
import os
import pandas as pd
import numpy as np
import json
import pickle
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------- Config Constants --------------
SEED = 42
TEST_SIZE = 0.2
# ----------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train headline classifier.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV data file.")
    parser.add_argument('--experiment_name', type=str, default="headline_classification", help="Experiment name.")
    parser.add_argument('--model', type=str, choices=['dense', 'lstm', 'random_forest', 'logistic_regression', 'all'], 
                        default='all', help="Which model to train.")
    parser.add_argument('--ratio', type=int, default=2, help="Downsampling ratio.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for NN models.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for NN models.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for NN models.")
    parser.add_argument('--embedding_model', type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence Transformer model name.")
    parser.add_argument('--max_vocab', type=int, default=10000, help="Maximum vocabulary size for LSTM.")
    parser.add_argument('--max_len', type=int, default=100, help="Maximum sequence length for LSTM.")
    parser.add_argument('--output_dir', type=str, default='./models/cat_classification', 
                    help="Directory to save trained models")
    parser.add_argument('--use_short_description', action='store_true', 
                        help="Use short_description in addition to headline.")
    args = parser.parse_args()
    return args


def save_confusion_matrix(y_true, y_pred, model_name, experiment_name, output_dir):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        experiment_name: Name of the experiment
        output_dir: Directory to save the output
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create directory path
    model_path = os.path.join(output_dir, experiment_name, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Generate heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure
    cm_path = os.path.join(model_path, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    return cm_path


def save_metrics(metrics, model_name, experiment_name, output_dir):
    """Save metrics to a JSON file in the experiment directory."""
    # Create directory path
    model_path = os.path.join(output_dir, experiment_name, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # Save metrics to JSON file
    metrics_path = os.path.join(model_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    return metrics_path


def save_model_to_exp_dir(model, model_name, experiment_name, output_dir):
    """Save model to a directory with experiment name as subdirectory."""
    # Create directory path with experiment name inside output directory
    save_path = os.path.join(output_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Create model-specific directory
    model_path = os.path.join(save_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # Save model in the directory
    if model_name.startswith(('random_forest', 'logistic_regression')):
        # For sklearn models
        model_save_path = os.path.join(model_path, f"{model_name}.pkl")
        joblib.dump(model, model_save_path)
    else:
        # For keras models
        model_save_path = os.path.join(model_path, f"{model_name}.keras")
        model.save(model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    return model_save_path


class MetricsHistory(Callback):
    def __init__(self, validation_data=None):
        super(MetricsHistory, self).__init__()
        self.validation_data = validation_data
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))


def downsample(df, ratio=2):
    """Downsample the majority class to balance the dataset."""
    majority = df[df['category'] == 0]
    minority = df[df['category'] == 1]
    
    print(f"Original class distribution: Class 0: {len(majority)}, Class 1: {len(minority)}")
    
    majority_downsampled = resample(
        majority, replace=False,
        n_samples=ratio * len(minority),
        random_state=SEED
    )
    df_downsampled = pd.concat([majority_downsampled, minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Downsampled class distribution: Class 0: {len(df_downsampled[df_downsampled['category'] == 0])}, "
          f"Class 1: {len(df_downsampled[df_downsampled['category'] == 1])}")
    
    return df_downsampled


def prepare_text_features(df, use_short_description=False):
    """Prepare text features from the dataframe."""
    if use_short_description:
        # Combine headline and short_description
        texts = df['headline'] + " " + df['short_description'].fillna("")
    else:
        texts = df['headline']
    
    return texts.tolist(), df['category'].values


def embed_text(texts, model_name):
    """Generate embeddings using a pre-trained Sentence Transformer model."""
    print(f"Generating embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def preprocess_text_lstm(texts, max_vocab=10000, max_len=100):
    """Tokenize and pad text for LSTM model."""
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer


def train_random_forest(X_train, y_train, X_val, y_val, args):
    """Train a Random Forest classifier."""
    print("Training Random Forest classifier...")
    
    # Parameters
    params = {"model_type": "RandomForest"}
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    
    metrics = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_f1": float(val_f1),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "parameters": params
    }
    
    # Save model to directory
    save_model_to_exp_dir(clf, "random_forest", args.experiment_name, args.output_dir)
    
    # Save metrics to file
    save_metrics(metrics, "random_forest", args.experiment_name, args.output_dir)

    # Generate test predictions and confusion matrix
    y_pred = clf.predict(X_val)
    save_confusion_matrix(y_val, y_pred, "random_forest", args.experiment_name, args.output_dir)
    
    print(f"Random Forest - Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    return val_acc, val_f1


def train_logistic_regression(X_train, y_train, X_val, y_val, args):
    """Train a Logistic Regression classifier."""
    print("Training Logistic Regression classifier...")
    
    # Parameters
    params = {"model_type": "LogisticRegression"}
    
    # Train model
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    
    metrics = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_f1": float(val_f1),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "parameters": params
    }
    
    # Save model to directory
    save_model_to_exp_dir(clf, "logistic_regression", args.experiment_name, args.output_dir)
    
    # Save metrics to file
    save_metrics(metrics, "logistic_regression", args.experiment_name, args.output_dir)

    # Generate test predictions and confusion matrix
    y_pred = clf.predict(X_val)
    save_confusion_matrix(y_val, y_pred, "logistic_regression", args.experiment_name, args.output_dir)
    
    print(f"Logistic Regression - Validation Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    return val_acc, val_f1


def train_dense_nn(X_train, y_train, X_val, y_val, input_dim, epochs, batch_size, lr, args):
    """Train a Dense Neural Network."""
    print("Training Dense Neural Network...")
    
    # Parameters
    params = {
        "model_type": "DenseNN",
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    }
    
    # Define model
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Create metrics history callback
    metrics_history = MetricsHistory()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, metrics_history],
        verbose=2
    )
    
    # Calculate metrics
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    
    # Calculate additional metrics
    val_preds = (model.predict(X_val) > 0.5).astype(int)
    val_f1 = f1_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    
    metrics = {
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss),
        "val_f1": float(val_f1),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "parameters": params,
        "training_history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    
    # Save model to directory
    save_model_to_exp_dir(model, "dense_nn", args.experiment_name, args.output_dir)
    
    # Save metrics to file
    save_metrics(metrics, "dense_nn", args.experiment_name, args.output_dir)

    # Generate test predictions and confusion matrix
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    save_confusion_matrix(y_val, y_pred, "dense_nn", args.experiment_name, args.output_dir)
    
    print(f"Dense NN - Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    return val_accuracy, val_f1


def train_lstm(X_train_texts, y_train, X_val_texts, y_val, max_vocab, max_len, epochs, batch_size, lr, args):
    """Train an LSTM model."""
    print("Training LSTM model...")
    
    # Parameters
    params = {
        "model_type": "LSTM",
        "max_vocab": max_vocab,
        "max_len": max_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    }
    
    # Preprocess text
    X_train_seq, tokenizer = preprocess_text_lstm(X_train_texts, max_vocab, max_len)
    X_val_seq = tokenizer.texts_to_sequences(X_val_texts)
    X_val_seq = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    
    # Get actual vocabulary size
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)
    params["actual_vocab_size"] = vocab_size
    
    # Define model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=256, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Create metrics history callback
    metrics_history = MetricsHistory()
    
    # Train model
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, metrics_history],
        verbose=2
    )
    
    # Calculate metrics
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    
    # Calculate additional metrics
    val_preds = (model.predict(X_val_seq) > 0.5).astype(int)
    val_f1 = f1_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    
    metrics = {
        "val_accuracy": float(val_accuracy),
        "val_loss": float(val_loss),
        "val_f1": float(val_f1),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "parameters": params,
        "training_history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    
    # Save tokenizer for later use
    tokenizer_path = os.path.join(args.output_dir, args.experiment_name, "lstm", "tokenizer.pkl")
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save word index as JSON for reference
    word_index_path = os.path.join(args.output_dir, args.experiment_name, "lstm", "word_index.json")
    with open(word_index_path, 'w') as f:
        json.dump({"word_index": tokenizer.word_index}, f, indent=4)
    
    # Save model to directory
    save_model_to_exp_dir(model, "lstm", args.experiment_name, args.output_dir)
    
    # Save metrics to file
    save_metrics(metrics, "lstm", args.experiment_name, args.output_dir)

    # Generate test predictions and confusion matrix
    y_pred = (model.predict(X_val_seq) > 0.5).astype(int)
    save_confusion_matrix(y_val, y_pred, "lstm", args.experiment_name, args.output_dir)
    
    print(f"LSTM - Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    return val_accuracy, val_f1


def main():
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Parse arguments
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df['category'] = df['category'].astype(int)
    print(f"Original data shape: {df.shape}")
    
    # Downsample data
    df_downsampled = downsample(df, ratio=args.ratio)
    print(f"Downsampled data shape: {df_downsampled.shape}")
    
    # Prepare text features
    X_text, y = prepare_text_features(df_downsampled, args.use_short_description)
    
    # Track best model
    best_model = None
    best_f1 = 0.0
    
    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment parameters
    experiment_params = vars(args)
    with open(os.path.join(experiment_dir, 'experiment_params.json'), 'w') as f:
        json.dump(experiment_params, f, indent=4)
    
    # Train models based on user selection
    models_to_train = [args.model] if args.model != 'all' else ['random_forest', 'logistic_regression', 'dense', 'lstm']
    
    for model_type in models_to_train:
        if model_type in ['random_forest', 'logistic_regression', 'dense']:
            # Generate embeddings for these models
            X_embeddings = embed_text(X_text, args.embedding_model)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_embeddings, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
            )
            
            # Train selected model
            if model_type == 'random_forest':
                _, f1 = train_random_forest(X_train, y_train, X_val, y_val, args)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = 'random_forest'
                    
            elif model_type == 'logistic_regression':
                _, f1 = train_logistic_regression(X_train, y_train, X_val, y_val, args)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = 'logistic_regression'
                    
            elif model_type == 'dense':
                _, f1 = train_dense_nn(
                    X_train, y_train, X_val, y_val, 
                    input_dim=X_train.shape[1],
                    epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    lr=args.lr,
                    args=args
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = 'dense'
        
        elif model_type == 'lstm':
            # Split text data for LSTM
            X_train_texts, X_val_texts, y_train, y_val = train_test_split(
                X_text, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
            )
            
            # Train LSTM model
            _, f1 = train_lstm(
                X_train_texts, y_train, X_val_texts, y_val,
                max_vocab=args.max_vocab, 
                max_len=args.max_len,
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                lr=args.lr,
                args=args
            )
            if f1 > best_f1:
                best_f1 = f1
                best_model = 'lstm'
    
    # Save best model info
    best_model_info = {
        "best_model": best_model,
        "best_f1_score": float(best_f1)
    }
    with open(os.path.join(experiment_dir, 'best_model_info.json'), 'w') as f:
        json.dump(best_model_info, f, indent=4)
    
    print(f"\nTraining complete! Best model: {best_model} with F1 score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
