"""
Generate actionable insights from sentiment analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class ReviewInsightsGenerator:
    """Generate actionable insights from product reviews"""
    
    def __init__(self):
        self.insights = {}
        
    def analyze_sentiment_trends(self, reviews_df: pd.DataFrame, 
                                date_column: str = 'date',
                                sentiment_column: str = 'sentiment') -> Dict:
        """Analyze sentiment trends over time"""
        
        if date_column not in reviews_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in dataframe")
        
        # Convert date column to datetime
        reviews_df['date'] = pd.to_datetime(reviews_df[date_column])
        
        # Group by time periods
        insights = {}
        
        # Daily trends
        daily_trends = reviews_df.groupby(
            reviews_df['date'].dt.date
        )[sentiment_column].value_counts(normalize=True).unstack().fillna(0)
        
        # Weekly trends
        reviews_df['week'] = reviews_df['date'].dt.isocalendar().week
        reviews_df['year'] = reviews_df['date'].dt.isocalendar().year
        weekly_trends = reviews_df.groupby(
            ['year', 'week']
        )[sentiment_column].value_counts(normalize=True).unstack().fillna(0)
        
        # Monthly trends
        reviews_df['month'] = reviews_df['date'].dt.month
        monthly_trends = reviews_df.groupby(
            ['year', 'month']
        )[sentiment_column].value_counts(normalize=True).unstack().fillna(0)
        
        insights['daily_trends'] = daily_trends
        insights['weekly_trends'] = weekly_trends
        insights['monthly_trends'] = monthly_trends
        
        # Calculate key metrics
        total_reviews = len(reviews_df)
        positive_reviews = len(reviews_df[reviews_df[sentiment_column] == 'positive'])
        negative_reviews = len(reviews_df[reviews_df[sentiment_column] == 'negative'])
        neutral_reviews = len(reviews_df[reviews_df[sentiment_column] == 'neutral'])
        
        insights['summary'] = {
            'total_reviews': total_reviews,
            'positive_percentage': positive_reviews / total_reviews * 100,
            'negative_percentage': negative_reviews / total_reviews * 100,
            'neutral_percentage': neutral_reviews / total_reviews * 100,
            'sentiment_ratio': positive_reviews / max(negative_reviews, 1)
        }
        
        return insights
    
    def extract_key_issues(self, reviews_df: pd.DataFrame, 
                          text_column: str = 'text',
                          sentiment_column: str = 'sentiment',
                          product_column: str = 'product_id') -> Dict:
        """Extract key issues and pain points from reviews"""
        
        # Define issue categories and keywords
        issue_categories = {
            'quality': ['quality', 'durable', 'break', 'broken', 'damage', 'defect', 
                       'faulty', 'poor quality', 'cheap', 'flimsy', 'material'],
            'price': ['expensive', 'pricey', 'overpriced', 'cheap', 'value', 
                     'worth', 'cost', 'price', 'affordable', 'budget'],
            'delivery': ['delivery', 'shipping', 'arrived', 'late', 'fast', 
                        'slow', 'package', 'packaging', 'damaged'],
            'customer_service': ['service', 'support', 'help', 'rude', 'friendly',
                                'response', 'customer service', 'representative'],
            'functionality': ['work', 'function', 'easy', 'difficult', 'use',
                            'complicated', 'simple', 'feature', 'performance'],
            'expectations': ['expect', 'disappoint', 'expectation', 'promise',
                           'description', 'photo', 'advertise', 'match']
        }
        
        insights = {}
        
        # Analyze negative reviews for issues
        negative_reviews = reviews_df[reviews_df[sentiment_column] == 'negative']
        
        if len(negative_reviews) > 0:
            issue_counts = {category: 0 for category in issue_categories}
            issue_examples = {category: [] for category in issue_categories}
            
            for _, review in negative_reviews.iterrows():
                text = str(review[text_column]).lower()
                
                for category, keywords in issue_categories.items():
                    for keyword in keywords:
                        if keyword in text:
                            issue_counts[category] += 1
                            if len(issue_examples[category]) < 3:  # Keep 3 examples max
                                issue_examples[category].append({
                                    'text': review[text_column][:200] + '...',
                                    'product': review.get(product_column, 'N/A')
                                })
                            break  # Count only once per category per review
            
            # Sort issues by frequency
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            insights['top_issues'] = [
                {
                    'issue': issue[0],
                    'count': issue[1],
                    'percentage': issue[1] / len(negative_reviews) * 100,
                    'examples': issue_examples[issue[0]]
                }
                for issue in sorted_issues if issue[1] > 0
            ]
            
            # Identify most problematic products
            if product_column in reviews_df.columns:
                product_issues = negative_reviews.groupby(product_column).size()
                insights['problematic_products'] = product_issues.nlargest(10).to_dict()
        
        return insights
    
    def analyze_positive_aspects(self, reviews_df: pd.DataFrame,
                               text_column: str = 'text',
                               sentiment_column: str = 'sentiment') -> Dict:
        """Analyze positive aspects from reviews"""
        
        positive_keywords = {
            'quality': ['excellent quality', 'high quality', 'durable', 'well made',
                       'solid', 'premium', 'sturdy', 'reliable'],
            'value': ['great value', 'good value', 'worth it', 'affordable',
                     'reasonable price', 'bargain', 'deal'],
            'experience': ['love', 'happy', 'satisfied', 'pleased', 'impressed',
                          'amazing', 'awesome', 'fantastic', 'wonderful'],
            'features': ['easy to use', 'user friendly', 'convenient', 'functional',
                        'feature rich', 'versatile', 'practical'],
            'service': ['great service', 'helpful', 'responsive', 'friendly',
                       'professional', 'quick response', 'excellent support']
        }
        
        positive_reviews = reviews_df[reviews_df[sentiment_column] == 'positive']
        
        insights = {}
        
        if len(positive_reviews) > 0:
            aspect_counts = {aspect: 0 for aspect in positive_keywords}
            aspect_examples = {aspect: [] for aspect in positive_keywords}
            
            for _, review in positive_reviews.iterrows():
                text = str(review[text_column]).lower()
                
                for aspect, keywords in positive_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            aspect_counts[aspect] += 1
                            if len(aspect_examples[aspect]) < 3:
                                aspect_examples[aspect].append(
                                    review[text_column][:200] + '...'
                                )
                            break
            
            # Sort positive aspects by frequency
            sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
            
            insights['top_positive_aspects'] = [
                {
                    'aspect': aspect[0],
                    'count': aspect[1],
                    'percentage': aspect[1] / len(positive_reviews) * 100,
                    'examples': aspect_examples[aspect[0]]
                }
                for aspect in sorted_aspects if aspect[1] > 0
            ]
        
        return insights
    
    def generate_seller_recommendations(self, insights: Dict) -> List[Dict]:
        """Generate actionable recommendations for sellers"""
        
        recommendations = []
        
        # Check for quality issues
        if 'top_issues' in insights:
            for issue in insights['top_issues']:
                if issue['percentage'] > 10:  # If more than 10% of negative reviews mention this
                    rec = {
                        'priority': 'High' if issue['percentage'] > 20 else 'Medium',
                        'issue': issue['issue'].title(),
                        'impact': f"{issue['percentage']:.1f}% of negative reviews",
                        'recommendation': self._get_recommendation_for_issue(issue['issue']),
                        'expected_outcome': self._get_expected_outcome(issue['issue'])
                    }
                    recommendations.append(rec)
        
        # Check for positive aspects to reinforce
        if 'top_positive_aspects' in insights and insights['top_positive_aspects']:
            top_aspect = insights['top_positive_aspects'][0]
            rec = {
                'priority': 'Low',
                'issue': f"Strengthen {top_aspect['aspect']}",
                'impact': f"{top_aspect['percentage']:.1f}% of positive reviews mention this",
                'recommendation': f"Highlight {top_aspect['aspect']} in product descriptions and marketing",
                'expected_outcome': "Increased customer satisfaction and conversion rates"
            }
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def _get_recommendation_for_issue(self, issue: str) -> str:
        """Get specific recommendation for each issue type"""
        
        recommendations = {
            'quality': "Implement stricter quality control checks and consider supplier audit. "
                      "Add product testing before shipping.",
            'price': "Review pricing strategy. Consider bundle offers, loyalty discounts, "
                    "or value-added services.",
            'delivery': "Optimize shipping partners. Provide better packaging. "
                       "Offer multiple shipping options with clear delivery estimates.",
            'customer_service': "Train customer service team on empathy and problem-solving. "
                              "Implement faster response time targets and escalation procedures.",
            'functionality': "Improve product documentation. Create tutorial videos. "
                           "Consider product redesign for better usability.",
            'expectations': "Update product descriptions with accurate specifications. "
                          "Use real customer photos. Set realistic expectations."
        }
        
        return recommendations.get(issue, "Review customer feedback for specific improvements.")
    
    def _get_expected_outcome(self, issue: str) -> str:
        """Get expected outcome for addressing each issue"""
        
        outcomes = {
            'quality': "Reduced returns, increased customer satisfaction, better reviews",
            'price': "Improved perceived value, increased sales, better competitiveness",
            'delivery': "Fewer shipping complaints, higher customer trust, repeat purchases",
            'customer_service': "Improved customer retention, positive word-of-mouth, higher NPS",
            'functionality': "Reduced support tickets, better user experience, higher ratings",
            'expectations': "Fewer disappointment complaints, better product-match, higher satisfaction"
        }
        
        return outcomes.get(issue, "Improved customer experience and business metrics")
    
    def generate_customer_experience_report(self, reviews_df: pd.DataFrame,
                                          sentiment_column: str = 'sentiment') -> Dict:
        """Generate comprehensive customer experience report"""
        
        report = {}
        
        # Overall sentiment metrics
        sentiment_dist = reviews_df[sentiment_column].value_counts(normalize=True) * 100
        
        report['overall_metrics'] = {
            'total_reviews': len(reviews_df),
            'sentiment_distribution': sentiment_dist.to_dict(),
            'net_sentiment_score': (
                sentiment_dist.get('positive', 0) - sentiment_dist.get('negative', 0)
            )
        }
        
        # Calculate response rate (if response data available)
        if 'response' in reviews_df.columns:
            response_rate = reviews_df['response'].notna().mean() * 100
            report['response_metrics'] = {
                'response_rate': response_rate,
                'avg_response_time_days': self._calculate_avg_response_time(reviews_df)
            }
        
        # Identify sentiment drivers
        if 'rating' in reviews_df.columns:
            rating_sentiment_corr = reviews_df.groupby('rating')[sentiment_column].value_counts(normalize=True)
            report['rating_analysis'] = {
                'avg_rating': reviews_df['rating'].mean(),
                'rating_sentiment_correlation': rating_sentiment_corr.unstack().fillna(0).to_dict()
            }
        
        # Generate improvement targets
        report['improvement_targets'] = self._generate_improvement_targets(report)
        
        return report
    
    def _calculate_avg_response_time(self, reviews_df: pd.DataFrame) -> float:
        """Calculate average response time in days"""
        if all(col in reviews_df.columns for col in ['review_date', 'response_date']):
            try:
                review_dates = pd.to_datetime(reviews_df['review_date'])
                response_dates = pd.to_datetime(reviews_df['response_date'])
                response_times = (response_dates - review_dates).dt.days
                return response_times.mean()
            except:
                return None
        return None
    
    def _generate_improvement_targets(self, report: Dict) -> Dict:
        """Generate SMART improvement targets"""
        
        current_positive = report['overall_metrics']['sentiment_distribution'].get('positive', 0)
        current_negative = report['overall_metrics']['sentiment_distribution'].get('negative', 0)
        
        targets = {
            'short_term': {
                'goal': f"Increase positive sentiment from {current_positive:.1f}% to {min(85, current_positive + 5):.1f}%",
                'timeframe': '3 months',
                'actions': ['Address top 3 customer issues', 'Improve response rate to >80%']
            },
            'medium_term': {
                'goal': f"Reduce negative sentiment from {current_negative:.1f}% to {max(5, current_negative - 5):.1f}%",
                'timeframe': '6 months',
                'actions': ['Implement quality control improvements', 'Optimize pricing strategy']
            },
            'long_term': {
                'goal': f"Achieve net sentiment score of +40 (Currently: {report['overall_metrics']['net_sentiment_score']:.1f})",
                'timeframe': '1 year',
                'actions': ['Establish brand reputation for quality', 'Build customer loyalty program']
            }
        }
        
        return targets
    
    def export_insights_report(self, insights: Dict, filename: str = None):
        """Export insights as a comprehensive report"""
        
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"review_insights_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"✓ Insights report exported to {filename}")
        
        return filename