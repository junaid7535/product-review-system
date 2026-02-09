"""
Generate synthetic product reviews for development and testing
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import random
from datetime import datetime, timedelta

class SyntheticReviewGenerator:
    """Generate realistic synthetic product reviews"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Templates for different sentiment categories
        self.templates = {
            'positive': [
                "Absolutely love this {product}! The {feature} is fantastic and it works perfectly.",
                "Great value for money. The {product} exceeded my expectations in every way.",
                "Excellent quality and fast shipping. Would definitely recommend this {product} to friends.",
                "Very happy with my purchase. The {product} is exactly as described and works great.",
                "Best {product} I've ever bought! The {feature} is amazing and the quality is top-notch.",
                "Perfect! The {product} arrived quickly and works flawlessly. Very satisfied customer.",
                "Impressed with the quality of this {product}. The {feature} works exactly as expected.",
                "Highly recommended! Great {product} at a reasonable price. Will buy again.",
                "Exactly what I needed. The {product} is durable and performs well. Very pleased.",
                "Outstanding product! The {feature} is innovative and the overall quality is superb."
            ],
            'negative': [
                "Very disappointed with this {product}. The {feature} stopped working after a week.",
                "Poor quality product. The {product} broke easily and customer service was unhelpful.",
                "Not worth the money. The {product} doesn't work as advertised and feels cheap.",
                "Had issues with delivery and the {product} arrived damaged. Very frustrating experience.",
                "The {product} is not what I expected. The {feature} doesn't work properly.",
                "Waste of money. The {product} stopped working after minimal use. Avoid this.",
                "Terrible quality. The {product} arrived with defects and the return process was difficult.",
                "Very unhappy with this purchase. The {product} failed to meet basic expectations.",
                "The {feature} is poorly designed and the overall {product} feels flimsy.",
                "Would not recommend. The {product} has multiple issues and support was unresponsive."
            ],
            'neutral': [
                "The {product} is okay. It works but nothing special. The {feature} is average.",
                "Decent product for the price. The {product} does what it's supposed to do.",
                "Average quality {product}. It serves its purpose but could be better.",
                "The {product} is fine. Not great, not terrible. The {feature} works as expected.",
                "It's an acceptable {product}. Does the job but I've seen better.",
                "Standard {product}. Works adequately but lacks any standout features.",
                "The {product} meets basic requirements. Nothing exceptional to report.",
                "Functional {product}. It works but don't expect anything impressive.",
                "An average purchase. The {product} performs adequately for daily use.",
                "The {product} is what you'd expect. No major issues but no pleasant surprises either."
            ]
        }
        
        # Products and features
        self.products = [
            ('smartphone', ['camera', 'battery life', 'screen', 'performance', 'design']),
            ('laptop', ['keyboard', 'display', 'battery', 'speed', 'build quality']),
            ('headphones', ['sound quality', 'comfort', 'battery', 'noise cancellation', 'build']),
            ('smartwatch', ['battery', 'display', 'fitness tracking', 'apps', 'design']),
            ('tablet', ['screen', 'performance', 'battery', 'portability', 'stylus']),
            ('blender', ['power', 'noise level', 'ease of cleaning', 'durability', 'settings']),
            ('coffee maker', ['brew quality', 'speed', 'ease of use', 'cleanup', 'features']),
            ('vacuum cleaner', ['suction', 'noise', 'maneuverability', 'dust capacity', 'attachments'])
        ]
        
        # Seller names
        self.sellers = [
            'TechGadgets Inc.', 'HomeEssentials Co.', 'Electronics Hub', 
            'Quality Goods Ltd.', 'Innovation Store', 'Smart Living', 
            'Premium Brands', 'Value Deals'
        ]
        
        # Common issues for negative reviews
        self.common_issues = [
            'arrived damaged', 'stopped working quickly', 'poor battery life',
            'difficult to assemble', 'missing parts', 'defective component',
            'slow performance', 'overheating issues', 'software problems'
        ]
    
    def generate_reviews(self, n_reviews: int = 10000, 
                        sentiment_dist: Dict = None) -> pd.DataFrame:
        """Generate synthetic product reviews"""
        
        if sentiment_dist is None:
            sentiment_dist = {'positive': 0.6, 'negative': 0.25, 'neutral': 0.15}
        
        reviews = []
        dates = []
        ratings = []
        sentiments = []
        product_categories = []
        sellers_list = []
        
        # Generate dates over the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for i in range(n_reviews):
            # Select sentiment based on distribution
            sentiment = np.random.choice(
                list(sentiment_dist.keys()),
                p=list(sentiment_dist.values())
            )
            
            # Select product and feature
            product_idx = random.randint(0, len(self.products) - 1)
            product_name, features = self.products[product_idx]
            feature = random.choice(features)
            
            # Select template and generate review
            template = random.choice(self.templates[sentiment])
            review_text = template.format(product=product_name, feature=feature)
            
            # Add specific details for negative reviews
            if sentiment == 'negative' and random.random() < 0.7:
                issue = random.choice(self.common_issues)
                review_text += f" Specifically, it {issue}."
            
            # Generate rating (1-5 stars)
            if sentiment == 'positive':
                rating = random.randint(4, 5)
            elif sentiment == 'negative':
                rating = random.randint(1, 2)
            else:
                rating = random.randint(3, 4)
            
            # Generate random date
            random_days = random.randint(0, 365)
            review_date = start_date + timedelta(days=random_days)
            
            # Select seller
            seller = random.choice(self.sellers)
            
            # Sometimes add a seller response (more likely for negative reviews)
            response = None
            if sentiment == 'negative' and random.random() < 0.4:
                response_date = review_date + timedelta(days=random.randint(1, 7))
                response = self._generate_seller_response(product_name, sentiment)
            
            reviews.append(review_text)
            dates.append(review_date)
            ratings.append(rating)
            sentiments.append(sentiment)
            product_categories.append(product_name)
            sellers_list.append(seller)
        
        # Create dataframe
        df = pd.DataFrame({
            'review_id': [f'REV_{i:06d}' for i in range(n_reviews)],
            'product': product_categories,
            'seller': sellers_list,
            'review_text': reviews,
            'rating': ratings,
            'sentiment': sentiments,
            'review_date': dates,
            'verified_purchase': np.random.choice([True, False], n_reviews, p=[0.8, 0.2]),
            'helpful_votes': np.random.poisson(3, n_reviews),
            'response': [self._generate_seller_response if random.random() < 0.3 else None 
                        for _ in range(n_reviews)]
        })
        
        # Sort by date
        df = df.sort_values('review_date').reset_index(drop=True)
        
        print(f"Generated {n_reviews} synthetic reviews")
        print(f"Sentiment distribution: {df['sentiment'].value_counts(normalize=True).to_dict()}")
        
        return df
    
    def _generate_seller_response(self, product: str, sentiment: str) -> str:
        """Generate realistic seller response"""
        
        if sentiment == 'negative':
            responses = [
                f"We're sorry to hear about your experience with our {product}. "
                "Please contact our customer service team so we can make this right.",
                f"Thank you for your feedback about our {product}. We take quality seriously "
                "and would like to investigate this issue further. Please reach out to us.",
                f"We apologize for the inconvenience with your {product}. Our team is committed "
                "to resolving this and ensuring your satisfaction."
            ]
        else:
            responses = [
                f"Thank you for your positive review of our {product}! We're glad you're happy "
                "with your purchase.",
                f"We appreciate your feedback on our {product}. Thank you for choosing us!",
                f"Thank you for taking the time to review our {product}. We're delighted "
                "that you're satisfied with your purchase."
            ]
        
        return random.choice(responses)
    
    def generate_with_metadata(self, n_reviews: int = 10000) -> pd.DataFrame:
        """Generate reviews with additional metadata"""
        
        base_df = self.generate_reviews(n_reviews)
        
        # Add additional features
        base_df['review_length'] = base_df['review_text'].apply(len)
        base_df['word_count'] = base_df['review_text'].apply(lambda x: len(str(x).split()))
        
        # Add product categories
        category_map = {
            'smartphone': 'Electronics',
            'laptop': 'Electronics',
            'headphones': 'Electronics',
            'smartwatch': 'Electronics',
            'tablet': 'Electronics',
            'blender': 'Home & Kitchen',
            'coffee maker': 'Home & Kitchen',
            'vacuum cleaner': 'Home & Kitchen'
        }
        base_df['category'] = base_df['product'].map(category_map)
        
        # Add price ranges
        price_ranges = {
            'smartphone': (300, 1200),
            'laptop': (500, 2000),
            'headphones': (50, 400),
            'smartwatch': (150, 800),
            'tablet': (200, 1000),
            'blender': (30, 200),
            'coffee maker': (40, 300),
            'vacuum cleaner': (80, 500)
        }
        
        base_df['price'] = base_df.apply(
            lambda row: random.uniform(*price_ranges[row['product']]), axis=1
        )
        
        return base_df