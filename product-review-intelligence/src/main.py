"""
Main execution script for Product Review Intelligence System
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from data.review_generator import SyntheticReviewGenerator
from nlp.text_preprocessor import ReviewPreprocessor
from nlp.feature_extractor import ReviewFeatureExtractor
from models.model_trainer import SentimentModelTrainer
from evaluation.insights_generator import ReviewInsightsGenerator
from evaluation.visualizations import ReviewVisualizer


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("PRODUCT REVIEW INTELLIGENCE SYSTEM")
    print("=" * 70)
    
    # Step 1: Generate synthetic reviews
    print("\n1. GENERATING SYNTHETIC PRODUCT REVIEWS")
    print("-" * 40)
    
    generator = SyntheticReviewGenerator(seed=42)
    reviews_df = generator.generate_with_metadata(n_reviews=10000)
    
    print(f"Generated {len(reviews_df)} reviews")
    print(f"Sentiment distribution:")
    print(reviews_df['sentiment'].value_counts())
    
    # Step 2: Preprocess text data
    print("\n2. PREPROCESSING REVIEW TEXT")
    print("-" * 40)
    
    preprocessor = ReviewPreprocessor(use_spacy=True)
    print("Sample preprocessing:")
    
    sample_review = reviews_df['review_text'].iloc[0]
    print(f"Original: {sample_review[:100]}...")
    processed = preprocessor.preprocess_text(sample_review)
    print(f"Processed: {processed[:100]}...")
    
    # Step 3: Initialize model trainer
    print("\n3. INITIALIZING SENTIMENT MODEL TRAINER")
    print("-" * 40)
    
    trainer = SentimentModelTrainer(
        test_size=0.2,
        cv_folds=5,
        random_state=42
    )
    
    # Step 4: Prepare data for training
    print("\n4. PREPARING DATA FOR TRAINING")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        reviews_df['review_text'].tolist(),
        reviews_df['sentiment'].tolist()
    )
    
    # Step 5: Train models
    print("\n5. TRAINING SENTIMENT CLASSIFICATION MODELS")
    print("-" * 40)
    
    results = trainer.train_models(X_train, y_train)
    
    # Step 6: Select best model
    print("\n6. SELECTING BEST MODEL")
    print("-" * 40)
    
    best_model, best_model_name = trainer.select_best_model()
    
    # Step 7: Evaluate on test set
    print("\n7. EVALUATING ON TEST SET")
    print("-" * 40)
    
    test_metrics = trainer.evaluate_on_test(X_test, y_test)
    
    # Step 8: Generate insights from reviews
    print("\n8. GENERATING ACTIONABLE INSIGHTS")
    print("-" * 40)
    
    insights_generator = ReviewInsightsGenerator()
    
    # Analyze sentiment trends
    print("\nAnalyzing sentiment trends...")
    sentiment_insights = insights_generator.analyze_sentiment_trends(
        reviews_df, 
        date_column='review_date',
        sentiment_column='sentiment'
    )
    
    # Extract key issues
    print("Extracting key issues from negative reviews...")
    issue_insights = insights_generator.extract_key_issues(
        reviews_df,
        text_column='review_text',
        sentiment_column='sentiment',
        product_column='product'
    )
    
    # Analyze positive aspects
    print("Analyzing positive aspects...")
    positive_insights = insights_generator.analyze_positive_aspects(
        reviews_df,
        text_column='review_text',
        sentiment_column='sentiment'
    )
    
    # Generate recommendations
    print("Generating seller recommendations...")
    all_insights = {
        'sentiment_trends': sentiment_insights,
        'key_issues': issue_insights,
        'positive_aspects': positive_insights
    }
    
    recommendations = insights_generator.generate_seller_recommendations(all_insights)
    
    # Step 9: Generate customer experience report
    print("\n9. GENERATING CUSTOMER EXPERIENCE REPORT")
    print("-" * 40)
    
    customer_experience_report = insights_generator.generate_customer_experience_report(
        reviews_df,
        sentiment_column='sentiment'
    )
    
    # Step 10: Save model and insights
    print("\n10. SAVING MODEL AND INSIGHTS")
    print("-" * 40)
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / "models" / "sentiment_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    
    metadata = {
        'model_name': best_model_name,
        'performance': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score']
        },
        'dataset_info': {
            'total_reviews': len(reviews_df),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }
    
    trainer.save_model(str(model_path), metadata)
    
    # Save insights report
    insights_path = artifacts_dir / "reports" / "review_insights.json"
    insights_path.parent.mkdir(exist_ok=True)
    
    full_insights = {
        'model_performance': test_metrics,
        'sentiment_analysis': sentiment_insights,
        'customer_issues': issue_insights,
        'positive_aspects': positive_insights,
        'recommendations': recommendations,
        'customer_experience': customer_experience_report
    }
    
    insights_generator.export_insights_report(full_insights, str(insights_path))
    
    # Step 11: Generate visualizations
    print("\n11. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    visualizer = ReviewVisualizer()
    
    # Plot sentiment distribution
    visualizer.plot_sentiment_distribution(reviews_df['sentiment'])
    
    # Plot sentiment trends over time
    if 'sentiment_trends' in full_insights and 'monthly_trends' in full_insights['sentiment_trends']:
        visualizer.plot_sentiment_trends(full_insights['sentiment_trends']['monthly_trends'])
    
    # Plot key issues
    if 'key_issues' in full_insights and 'top_issues' in full_insights['key_issues']:
        visualizer.plot_key_issues(full_insights['key_issues']['top_issues'])
    
    # Step 12: Print summary
    print("\n" + "=" * 70)
    print("PROJECT SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Reviews: {len(reviews_df):,}")
    print(f"   Positive: {reviews_df['sentiment'].value_counts().get('positive', 0):,}")
    print(f"   Negative: {reviews_df['sentiment'].value_counts().get('negative', 0):,}")
    print(f"   Neutral: {reviews_df['sentiment'].value_counts().get('neutral', 0):,}")
    
    print(f"\n🤖 Model Performance:")
    print(f"   Best Model: {best_model_name}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    
    print(f"\n🔍 Key Insights Generated:")
    if 'key_issues' in full_insights and 'top_issues' in full_insights['key_issues']:
        for i, issue in enumerate(full_insights['key_issues']['top_issues'][:3], 1):
            print(f"   {i}. {issue['issue'].title()}: {issue['percentage']:.1f}% of negative reviews")
    
    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. [{rec['priority']}] {rec['recommendation'][:80]}...")
    
    print(f"\n💾 Saved Artifacts:")
    print(f"   Model: {model_path}")
    print(f"   Insights Report: {insights_path}")
    
    print("\n" + "=" * 70)
    print("✅ PRODUCT REVIEW INTELLIGENCE SYSTEM COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()