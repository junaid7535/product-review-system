"""
Advanced text preprocessing for product reviews
"""
import re
import string
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import emoji

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class ReviewPreprocessor:
    """Advanced preprocessing for product review text"""
    
    def __init__(self, language: str = 'english', use_spacy: bool = False):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("Downloading spaCy model...")
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
                self.nlp = spacy.load('en_core_web_sm')
        
        # Custom preprocessing patterns
        self.contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "won't": "will not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will"
        }
        
        # Product-specific terms to keep
        self.product_terms = {
            'price', 'quality', 'delivery', 'shipping', 'packaging',
            'product', 'item', 'order', 'customer', 'service',
            'return', 'refund', 'exchange', 'warranty', 'guarantee'
        }
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '😊': 'positive', '😃': 'positive', '😄': 'positive', '😁': 'positive',
            '😍': 'positive', '🥰': 'positive', '😘': 'positive', '😎': 'positive',
            '😢': 'negative', '😭': 'negative', '😞': 'negative', '😔': 'negative',
            '😠': 'negative', '😡': 'negative', '🤬': 'negative', '🤢': 'negative',
            '😐': 'neutral', '😑': 'neutral', '🙄': 'neutral', '🤔': 'neutral'
        }
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       lemmatize: bool = True,
                       handle_emojis: bool = True,
                       extract_features: bool = False) -> str:
        """
        Comprehensive text preprocessing pipeline
        
        Args:
            text: Input review text
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize tokens
            handle_emojis: Whether to handle emoji sentiment
            extract_features: Whether to return features dict
            
        Returns:
            Preprocessed text or features dictionary
        """
        if not isinstance(text, str):
            return "" if not extract_features else {}
        
        original_text = text
        
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Expand contractions
        text = self._expand_contractions(text)
        
        # 3. Handle emojis
        emoji_features = {}
        if handle_emojis:
            text, emoji_features = self._handle_emojis(text)
        
        # 4. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 5. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 6. Remove special characters and numbers (keep some punctuation for context)
        text = re.sub(r'[^a-zA-Z\s.!?]', ' ', text)
        
        # 7. Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Tokenize
        tokens = word_tokenize(text)
        
        # 9. Remove stopwords (but keep product-specific terms)
        if remove_stopwords:
            tokens = [
                token for token in tokens 
                if token not in self.stop_words or token in self.product_terms
            ]
        
        # 10. Lemmatization
        if lemmatize:
            if self.use_spacy:
                doc = self.nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc]
            else:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # 11. Join tokens back
        processed_text = ' '.join(tokens)
        
        if extract_features:
            features = {
                'processed_text': processed_text,
                'original_length': len(original_text),
                'processed_length': len(processed_text),
                'word_count': len(tokens),
                'unique_words': len(set(tokens)),
                'vocabulary_richness': len(set(tokens)) / len(tokens) if tokens else 0,
                'emoji_positive': emoji_features.get('positive', 0),
                'emoji_negative': emoji_features.get('negative', 0),
                'emoji_neutral': emoji_features.get('neutral', 0),
                'has_question': 1 if '?' in original_text else 0,
                'has_exclamation': 1 if '!' in original_text else 0,
                'sentiment_score': self._calculate_textblob_sentiment(original_text)
            }
            return features
        
        return processed_text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def _handle_emojis(self, text: str) -> tuple:
        """Extract and handle emoji sentiment"""
        emoji_count = {'positive': 0, 'negative': 0, 'neutral': 0}
        emojis_found = []
        
        # Extract emojis
        chars = list(text)
        emoji_indices = []
        
        for i, char in enumerate(chars):
            if char in emoji.UNICODE_EMOJI['en']:
                emojis_found.append(char)
                sentiment = self.emoji_sentiment.get(char, 'neutral')
                emoji_count[sentiment] += 1
                emoji_indices.append(i)
        
        # Replace emojis with sentiment tags
        text_without_emojis = ''.join([char for i, char in enumerate(chars) 
                                      if i not in emoji_indices])
        
        # Add sentiment tags
        if emoji_count['positive'] > emoji_count['negative']:
            text_without_emojis += " positive_emoji"
        elif emoji_count['negative'] > emoji_count['positive']:
            text_without_emojis += " negative_emoji"
        
        return text_without_emojis, emoji_count
    
    def _calculate_textblob_sentiment(self, text: str) -> float:
        """Calculate sentiment score using TextBlob"""
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess_text(text, **kwargs) for text in texts]
    
    def extract_text_features(self, texts: List[str]) -> Dict:
        """Extract comprehensive text features"""
        features = {
            'processed_texts': [],
            'text_features': []
        }
        
        for text in texts:
            result = self.preprocess_text(text, extract_features=True)
            if isinstance(result, dict):
                features['processed_texts'].append(result['processed_text'])
                features['text_features'].append({
                    k: v for k, v in result.items() 
                    if k != 'processed_text'
                })
            else:
                features['processed_texts'].append(result)
                features['text_features'].append({})
        
        return features