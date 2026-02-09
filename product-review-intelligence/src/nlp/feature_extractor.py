"""
Feature extraction from product reviews
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pandas as pd

class ReviewFeatureExtractor:
    """Extract features from preprocessed reviews"""
    
    def __init__(self, max_features: int = 5000, n_components: int = 10):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.word2vec_model = None
        
    def extract_tfidf_features(self, texts: List[str], 
                              ngram_range: Tuple = (1, 2),
                              use_idf: bool = True) -> np.ndarray:
        """Extract TF-IDF features"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=ngram_range,
                stop_words='english',
                use_idf=use_idf
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features
    
    def extract_bow_features(self, texts: List[str]) -> np.ndarray:
        """Extract Bag-of-Words features"""
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
            bow_features = self.count_vectorizer.fit_transform(texts)
        else:
            bow_features = self.count_vectorizer.transform(texts)
        
        return bow_features
    
    def extract_topic_features(self, texts: List[str], 
                              method: str = 'lda',
                              n_topics: int = 10) -> Tuple[np.ndarray, List]:
        """Extract topic modeling features"""
        
        # Create document-term matrix
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words='english'
            )
            doc_term_matrix = self.count_vectorizer.fit_transform(texts)
        else:
            doc_term_matrix = self.count_vectorizer.transform(texts)
        
        if method == 'lda':
            if self.lda_model is None:
                self.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    learning_method='online'
                )
                topic_features = self.lda_model.fit_transform(doc_term_matrix)
            else:
                topic_features = self.lda_model.transform(doc_term_matrix)
            
            # Get topic words
            feature_names = self.count_vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
        
        elif method == 'nmf':
            if self.nmf_model is None:
                self.nmf_model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    init='nndsvd'
                )
                topic_features = self.nmf_model.fit_transform(doc_term_matrix)
            else:
                topic_features = self.nmf_model.transform(doc_term_matrix)
            
            # Get topic words
            feature_names = self.count_vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(self.nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)
        
        else:
            raise ValueError("Method must be 'lda' or 'nmf'")
        
        return topic_features, topics
    
    def extract_sentiment_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentiment-based features"""
        from textblob import TextBlob
        
        features = []
        for text in texts:
            blob = TextBlob(text)
            
            sentiment_features = [
                blob.sentiment.polarity,  # Sentiment polarity (-1 to 1)
                blob.sentiment.subjectivity,  # Subjectivity (0 to 1)
                len(text.split()),  # Word count
                len(set(text.split())) / len(text.split()) if text.split() else 0,  # Lexical diversity
                text.count('!'),  # Exclamation count
                text.count('?'),  # Question count
                text.count('not'),  # Negation count
                sum(1 for word in text.split() if len(word) > 6) / len(text.split()) if text.split() else 0,  # Long word ratio
            ]
            features.append(sentiment_features)
        
        return np.array(features)
    
    def extract_readability_features(self, texts: List[str]) -> np.ndarray:
        """Extract readability and complexity features"""
        import syllables
        
        features = []
        
        for text in texts:
            words = text.split()
            sentences = text.split('.')
            
            if not words:
                features.append([0, 0, 0, 0, 0])
                continue
            
            # Calculate various readability metrics
            word_count = len(words)
            sentence_count = max(1, len(sentences))
            char_count = len(text)
            
            # Average sentence length
            avg_sentence_length = word_count / sentence_count
            
            # Average word length
            avg_word_length = char_count / word_count
            
            # Syllable count per word
            syllable_counts = [syllables.estimate(word) for word in words]
            avg_syllables = sum(syllable_counts) / word_count
            
            # Complex word ratio (words with 3+ syllables)
            complex_words = sum(1 for count in syllable_counts if count >= 3)
            complex_word_ratio = complex_words / word_count
            
            # Flesch Reading Ease Score approximation
            flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
            
            features.append([
                avg_sentence_length,
                avg_word_length,
                avg_syllables,
                complex_word_ratio,
                flesch_score
            ])
        
        return np.array(features)
    
    def extract_all_features(self, texts: List[str], 
                            include_tfidf: bool = True,
                            include_topics: bool = True,
                            include_sentiment: bool = True,
                            include_readability: bool = True) -> Dict:
        """Extract all available features"""
        
        features_dict = {}
        
        if include_tfidf:
            tfidf_features = self.extract_tfidf_features(texts)
            features_dict['tfidf'] = tfidf_features
        
        if include_topics:
            topic_features, topics = self.extract_topic_features(texts)
            features_dict['topics'] = topic_features
            features_dict['topic_words'] = topics
        
        if include_sentiment:
            sentiment_features = self.extract_sentiment_features(texts)
            features_dict['sentiment'] = sentiment_features
        
        if include_readability:
            readability_features = self.extract_readability_features(texts)
            features_dict['readability'] = readability_features
        
        # Combine all features
        all_features = []
        feature_names = []
        
        for feature_type, feature_matrix in features_dict.items():
            if feature_type not in ['topic_words']:
                if hasattr(feature_matrix, 'toarray'):
                    feature_matrix = feature_matrix.toarray()
                
                all_features.append(feature_matrix)
                
                # Generate feature names
                if feature_type == 'tfidf' and self.tfidf_vectorizer:
                    feature_names.extend(
                        [f'tfidf_{i}' for i in range(feature_matrix.shape[1])]
                    )
                elif feature_type == 'topics':
                    feature_names.extend(
                        [f'topic_{i}' for i in range(feature_matrix.shape[1])]
                    )
                elif feature_type == 'sentiment':
                    sentiment_names = ['polarity', 'subjectivity', 'word_count',
                                      'lexical_diversity', 'exclamations',
                                      'questions', 'negations', 'long_word_ratio']
                    feature_names.extend(sentiment_names)
                elif feature_type == 'readability':
                    readability_names = ['avg_sentence_len', 'avg_word_len',
                                        'avg_syllables', 'complex_word_ratio',
                                        'flesch_score']
                    feature_names.extend(readability_names)
        
        if all_features:
            combined_features = np.hstack(all_features)
            features_dict['combined'] = combined_features
            features_dict['feature_names'] = feature_names
        
        return features_dict
    
    def get_feature_importance(self, model, feature_type: str = 'tfidf'):
        """Get feature importance from trained model"""
        
        if feature_type == 'tfidf' and self.tfidf_vectorizer:
            if hasattr(model, 'coef_'):
                importance = model.coef_[0]
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                return sorted(zip(feature_names, importance), 
                            key=lambda x: abs(x[1]), reverse=True)
        
        return []