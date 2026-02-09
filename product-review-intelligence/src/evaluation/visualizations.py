"""
Visualizations for product review analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ReviewVisualizer:
    """Create visualizations for product review analysis"""
    
    def __init__(self, figsize: Tuple = (12, 8)):
        self.figsize = figsize
        
    def plot_sentiment_distribution(self, sentiment_series: pd.Series):
        """Plot sentiment distribution"""
        
        sentiment_counts = sentiment_series.value_counts()
        sentiment_percentages = sentiment_series.value_counts(normalize=True) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        bar_colors = [colors.get(sent, 'blue') for sent in sentiment_counts.index]
        
        ax1.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
        ax1.set_title('Sentiment Distribution (Count)', fontsize=14)
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        
        # Add value labels
        for i, (sent, count) in enumerate(sentiment_counts.items()):
            ax1.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
        
        # Percentage plot
        ax2.bar(sentiment_percentages.index, sentiment_percentages.values, color=bar_colors)
        ax2.set_title('Sentiment Distribution (%)', fontsize=14)
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Percentage')
        ax2.set_ylim([0, 100])
        
        # Add percentage labels
        for i, (sent, pct) in enumerate(sentiment_percentages.items()):
            ax2.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_trends(self, trends_df: pd.DataFrame):
        """Plot sentiment trends over time"""
        
        plt.figure(figsize=self.figsize)
        
        for column in trends_df.columns:
            plt.plot(trends_df.index, trends_df[column], marker='o', label=column, linewidth=2)
        
        plt.title('Sentiment Trends Over Time', fontsize=16)
        plt.xlabel('Time Period')
        plt.ylabel('Sentiment Percentage')
        plt.legend(title='Sentiment')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_key_issues(self, issues_data: List[Dict]):
        """Plot key issues from negative reviews"""
        
        if not issues_data:
            print("No issues data to plot")
            return
        
        issues = [issue['issue'] for issue in issues_data]
        percentages = [issue['percentage'] for issue in issues_data]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.8))
        
        # Use red gradient for negative issues
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(issues)))
        
        bars = ax.barh(issues, percentages, color=colors)
        ax.invert_yaxis()  # Highest percentage on top
        
        ax.set_xlabel('Percentage of Negative Reviews (%)')
        ax.set_title('Top Issues from Negative Reviews', fontsize=14)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_positive_aspects(self, aspects_data: List[Dict]):
        """Plot positive aspects from reviews"""
        
        if not aspects_data:
            print("No positive aspects data to plot")
            return
        
        aspects = [aspect['aspect'] for aspect in aspects_data]
        percentages = [aspect['percentage'] for aspect in aspects_data]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.8))
        
        # Use green gradient for positive aspects
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(aspects)))
        
        bars = ax.barh(aspects, percentages, color=colors)
        ax.invert_yaxis()
        
        ax.set_xlabel('Percentage of Positive Reviews (%)')
        ax.set_title('Top Positive Aspects', fontsize=14)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance_comparison(self, results: Dict):
        """Compare performance of different models"""
        
        models = list(results.keys())
        f1_scores = [results[model]['best_score'] for model in models 
                    if results[model].get('model') is not None]
        valid_models = [model for model in models 
                       if results[model].get('model') is not None]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(valid_models)))
        
        bars = ax.barh(valid_models, f1_scores, color=colors)
        
        ax.set_xlabel('F1 Score (Cross-Validation)')
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xlim([0, 1])
        
        # Add score labels
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_word_cloud(self, texts: List[str], 
                       sentiment: str = None,
                       max_words: int = 100):
        """Generate word cloud for reviews"""
        
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis' if sentiment == 'positive' else 'Reds' if sentiment == 'negative' else 'Blues',
            contour_width=1,
            contour_color='steelblue'
        ).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        title = 'Word Cloud'
        if sentiment:
            title += f' - {sentiment.capitalize()} Reviews'
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: List[str] = None):
        """Plot confusion matrix"""
        
        if class_names is None:
            class_names = ['Negative', 'Neutral', 'Positive']
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_rating_sentiment_correlation(self, reviews_df: pd.DataFrame):
        """Plot correlation between ratings and sentiment"""
        
        if 'rating' not in reviews_df.columns or 'sentiment' not in reviews_df.columns:
            print("Required columns not found in dataframe")
            return
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(reviews_df['rating'], reviews_df['sentiment'], 
                               normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        cross_tab.plot(kind='bar', ax=ax, stacked=True, 
                      color=['red', 'gray', 'green'])
        
        ax.set_title('Sentiment Distribution by Rating', fontsize=14)
        ax.set_xlabel('Rating (Stars)')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim([0, 100])
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_review_length_distribution(self, reviews_df: pd.DataFrame, 
                                       text_column: str = 'review_text'):
        """Plot distribution of review lengths"""
        
        review_lengths = reviews_df[text_column].apply(lambda x: len(str(x).split()))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(review_lengths, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Review Length Distribution', fontsize=14)
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.axvline(review_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {review_lengths.mean():.1f}')
        ax1.legend()
        
        # Box plot by sentiment
        if 'sentiment' in reviews_df.columns:
            sentiment_data = []
            sentiments = []
            for sentiment in reviews_df['sentiment'].unique():
                lengths = review_lengths[reviews_df['sentiment'] == sentiment]
                sentiment_data.append(lengths)
                sentiments.append(sentiment)
            
            ax2.boxplot(sentiment_data, labels=sentiments)
            ax2.set_title('Review Length by Sentiment', fontsize=14)
            ax2.set_xlabel('Sentiment')
            ax2.set_ylabel('Number of Words')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_sentiment_dashboard(self, reviews_df: pd.DataFrame):
        """Create interactive dashboard for sentiment analysis"""
        
        if 'sentiment' not in reviews_df.columns or 'rating' not in reviews_df.columns:
            print("Required columns not found")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 
                           'Sentiment by Rating',
                           'Review Length Distribution',
                           'Sentiment Trends'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                  [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # 1. Pie chart for sentiment distribution
        sentiment_counts = reviews_df['sentiment'].value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, 
                  values=sentiment_counts.values,
                  hole=0.3,
                  marker_colors=['green', 'red', 'gray']),
            row=1, col=1
        )
        
        # 2. Bar chart for sentiment by rating
        cross_tab = pd.crosstab(reviews_df['rating'], reviews_df['sentiment'])
        for sentiment in cross_tab.columns:
            fig.add_trace(
                go.Bar(x=cross_tab.index, y=cross_tab[sentiment],
                      name=sentiment,
                      marker_color='green' if sentiment == 'positive' else 
                                  'red' if sentiment == 'negative' else 'gray'),
                row=1, col=2
            )
        
        # 3. Histogram for review length
        review_lengths = reviews_df['review_text'].apply(lambda x: len(str(x).split()))
        fig.add_trace(
            go.Histogram(x=review_lengths, nbinsx=30,
                        marker_color='blue',
                        opacity=0.7),
            row=2, col=1
        )
        
        # 4. Sentiment trends over time (if date column exists)
        if 'review_date' in reviews_df.columns:
            reviews_df['date'] = pd.to_datetime(reviews_df['review_date'])
            reviews_df['month'] = reviews_df['date'].dt.to_period('M').astype(str)
            
            monthly_sentiment = reviews_df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
            
            for sentiment in monthly_sentiment.columns:
                fig.add_trace(
                    go.Scatter(x=monthly_sentiment.index, 
                              y=monthly_sentiment[sentiment],
                              mode='lines+markers',
                              name=sentiment,
                              line=dict(width=2)),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Product Review Sentiment Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig