<<<<<<< HEAD
import os
import numpy as np
import configparser
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.text import Tokenizer   
#from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
load_model=tf.keras.models.load_model
Tokenizer=tf.keras.preprocessing.text.Tokenizer
import pickle
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = int(config["MODEL"]["max_len"])
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        model_path = config["MODEL"]["checkpoint_path"]
        tokenizer_path = os.path.join(config["MODEL"]["model_dir"], "tokenizer.pkl")
        
        try:
            # Check if model file exists
            if os.path.exists(model_path):
                # Try to load with GPU if available, otherwise use CPU
                try:
                    with tf.device('/GPU:0'):
                        self.model = load_model(model_path)
                        logger.info("Model loaded on GPU")
                except:
                    self.model = load_model(model_path)
                    logger.info("Model loaded on CPU")
                
                logger.info(f"Successfully loaded model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")
            
            # Load tokenizer if available
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
            else:
                logger.warning(f"Tokenizer file not found at {tokenizer_path}")
                
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        if not self.tokenizer:
            logger.error("Tokenizer not available")
            return None
            
        try:
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
            return padded
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return None
    
    def predict_sentiment(self, text):
        """Predict sentiment from text"""
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not available")
            return None, None
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            if processed_text is None:
                return None, None
            
            # Make prediction
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                prediction = self.model.predict(processed_text)
            
            # Get the predicted class and score
            predicted_class = np.argmax(prediction[0])
            confidence_score = float(prediction[0][predicted_class])
            
            sentiment_label = self.sentiment_labels[predicted_class]
            
            logger.info(f"Sentiment prediction: {sentiment_label} with confidence {confidence_score:.4f}")
            
            return sentiment_label, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
=======
import os
import numpy as np
import configparser
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.text import Tokenizer   
#from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
load_model=tf.keras.models.load_model
Tokenizer=tf.keras.preprocessing.text.Tokenizer
import pickle
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = int(config["MODEL"]["max_len"])
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        model_path = config["MODEL"]["checkpoint_path"]
        tokenizer_path = os.path.join(config["MODEL"]["model_dir"], "tokenizer.pkl")
        
        try:
            # Check if model file exists
            if os.path.exists(model_path):
                # Try to load with GPU if available, otherwise use CPU
                try:
                    with tf.device('/GPU:0'):
                        self.model = load_model(model_path)
                        logger.info("Model loaded on GPU")
                except:
                    self.model = load_model(model_path)
                    logger.info("Model loaded on CPU")
                
                logger.info(f"Successfully loaded model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")
            
            # Load tokenizer if available
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
            else:
                logger.warning(f"Tokenizer file not found at {tokenizer_path}")
                
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        if not self.tokenizer:
            logger.error("Tokenizer not available")
            return None
            
        try:
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
            return padded
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return None
    
    def predict_sentiment(self, text):
        """Predict sentiment from text"""
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not available")
            return None, None
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            if processed_text is None:
                return None, None
            
            # Make prediction
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                prediction = self.model.predict(processed_text)
            
            # Get the predicted class and score
            predicted_class = np.argmax(prediction[0])
            confidence_score = float(prediction[0][predicted_class])
            
            sentiment_label = self.sentiment_labels[predicted_class]
            
            logger.info(f"Sentiment prediction: {sentiment_label} with confidence {confidence_score:.4f}")
            
            return sentiment_label, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
>>>>>>> 865e3dee350745261eab842079e5aca439e51963
            return None, None