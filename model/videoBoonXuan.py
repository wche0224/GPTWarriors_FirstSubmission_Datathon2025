# Real-Time Trend Spotting Analysis
# Better suited for finding what's trending NOW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import re
from zipfile import ZipFile
from scipy import stats
import sys
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

class TrendSpotter:
    def __init__(self, videos_df):
        self.videos_df = videos_df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for trend analysis"""
        print("Preparing data for trend analysis...")
        
        # Convert publishedAt to datetime
        self.videos_df['publishedAt'] = pd.to_datetime(self.videos_df['publishedAt'], utc=True)
        
        # Create time windows
        now = self.videos_df['publishedAt'].max()
        self.videos_df['days_ago'] = (now - self.videos_df['publishedAt']).dt.days
        
        # Define time periods
        self.videos_df['time_period'] = pd.cut(self.videos_df['days_ago'], 
                                               bins=[-1, 7, 30, 90, 365, float('inf')],
                                               labels=['Last_7_days', 'Last_30_days', 'Last_90_days', 
                                                      'Last_year', 'Older'])
        
        # Calculate engagement metrics
        self.videos_df['engagement_rate'] = (
            self.videos_df['likeCount'] + self.videos_df['commentCount']
        ) / (self.videos_df['viewCount'] + 1)
        
        # Extract keywords from different sources
        self.extract_keywords()
    
    def extract_keywords(self):
        """Extract keywords from titles, descriptions, and tags"""
        print("Extracting keywords...")
        
        try:
            sw = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            sw = set(stopwords.words('english'))

        # compile once; matches numbers or words; case handled by lower()
        WORD_RE = re.compile(r'(?:^|\s)([1-9][0-9]*|[a-zA-Z]+)(?=\s|$)')

        def clean_and_extract(text):
            # text is a SINGLE string (a row's title/description), not the whole column
            s = '' if pd.isna(text) else str(text).lower()
            words = WORD_RE.findall(s)
            return [w for w in words if w and w not in sw]

        
        # Extract from titles
        self.videos_df['title_keywords'] = self.videos_df['title'].apply(clean_and_extract)
        
        # Extract from descriptions (first 200 chars to focus on main content)
        self.videos_df['desc_keywords'] = self.videos_df['description'].apply(
            lambda x: clean_and_extract(str(x)[:200]) if pd.notna(x) else []
        )
        
        # Extract from tags
        self.videos_df['tag_keywords'] = self.videos_df['tags'].apply(
            lambda x: [tag.strip().lower() for tag in str(x).split(',')] if pd.notna(x) else []
        )
    
    def find_trending_keywords(self, source='title', time_period='Last_7_days', min_videos=5):
        """Find trending keywords with momentum analysis"""
        
        keyword_col = f'{source}_keywords'
        
        # Filter by time period
        recent_videos = self.videos_df[self.videos_df['time_period'] == time_period]
        
        if len(recent_videos) < min_videos:
            print(f"Not enough videos in {time_period} period")
            return pd.DataFrame()
        
        # Count keywords in recent period
        recent_keyword_stats = defaultdict(lambda: {
            'count': 0, 'total_views': 0, 'total_likes': 0, 
            'total_comments': 0, 'videos': []
        })
        
        for idx, row in recent_videos.iterrows():
            keywords = row[keyword_col]
            for keyword in keywords:
                if len(keyword) >= 3:  # Filter short words
                    recent_keyword_stats[keyword]['count'] += 1
                    recent_keyword_stats[keyword]['total_views'] += row['viewCount']
                    recent_keyword_stats[keyword]['total_likes'] += row['likeCount']
                    recent_keyword_stats[keyword]['total_comments'] += row['commentCount']
                    recent_keyword_stats[keyword]['videos'].append(row['title'])
        
        # Convert to DataFrame
        trend_data = []
        for keyword, stats in recent_keyword_stats.items():
            if stats['count'] >= min_videos:  # Only keywords appearing in multiple videos
                avg_views = stats['total_views'] / stats['count']
                avg_engagement = (stats['total_likes'] + stats['total_comments']) / stats['count']
                
                trend_data.append({
                    'keyword': keyword,
                    'frequency': stats['count'],
                    'avg_views': avg_views,
                    'avg_engagement': avg_engagement,
                    'total_views': stats['total_views'],
                    'engagement_score': avg_views * (1 + avg_engagement/1000),  # Weighted score
                    'sample_videos': '; '.join(stats['videos'][:3])  # Show 3 example videos
                })
        
        df = pd.DataFrame(trend_data)
        return df.sort_values('engagement_score', ascending=False) if not df.empty else df
    
    def analyze_keyword_momentum(self, keyword, source='title'):
        """Analyze how a keyword's popularity changes over time"""
        
        keyword_col = f'{source}_keywords'
        
        # Find videos containing the keyword
        mask = self.videos_df[keyword_col].apply(lambda x: keyword.lower() in [k.lower() for k in x])
        keyword_videos = self.videos_df[mask]
        
        if len(keyword_videos) == 0:
            return None
        
        # Group by time periods
        momentum = keyword_videos.groupby('time_period').agg({
            'videoId': 'count',
            'viewCount': 'mean',
            'likeCount': 'mean',
            'commentCount': 'mean',
            'engagement_rate': 'mean'
        }).rename(columns={'videoId': 'video_count'})
        
        return momentum
    
    def get_trending_categories(self, time_period='Last_7_days'):
        """Analyze trending topic categories"""
        
        if 'topicCategories' not in self.videos_df.columns:
            return "No topic categories data available"
        
        recent_videos = self.videos_df[self.videos_df['time_period'] == time_period]
        
        # Extract categories
        category_stats = defaultdict(lambda: {'count': 0, 'total_engagement': 0})
        
        for idx, row in recent_videos.iterrows():
            if pd.notna(row['topicCategories']):
                categories = str(row['topicCategories']).split(',')
                for category in categories:
                    category = category.strip()
                    if category:
                        category_stats[category]['count'] += 1
                        category_stats[category]['total_engagement'] += row['engagement_rate']
        
        # Convert to DataFrame
        cat_data = []
        for category, stats in category_stats.items():
            if stats['count'] > 0:
                cat_data.append({
                    'category': category,
                    'video_count': stats['count'],
                    'avg_engagement': stats['total_engagement'] / stats['count']
                })
        
        return pd.DataFrame(cat_data).sort_values('avg_engagement', ascending=False)
    
    def detect_viral_patterns(self, time_period='Last_7_days'):
        """Detect patterns in viral content"""
        
        recent_videos = self.videos_df[self.videos_df['time_period'] == time_period]
        
        if len(recent_videos) == 0:
            return "No recent videos found"
        
        # Define viral threshold (top 10% by engagement)
        engagement_threshold = recent_videos['engagement_rate'].quantile(0.9)
        viral_videos = recent_videos[recent_videos['engagement_rate'] >= engagement_threshold]
        
        print(f"\n=== VIRAL CONTENT PATTERNS ({time_period}) ===")
        print(f"Total videos: {len(recent_videos)}")
        print(f"Viral videos (top 10%): {len(viral_videos)}")
        print(f"Viral threshold engagement rate: {engagement_threshold:.6f}")
        
        if len(viral_videos) > 0:
            # Analyze viral video characteristics
            print("\n--- Viral Video Characteristics ---")
            print(f"Average views: {viral_videos['viewCount'].mean():,.0f}")
            print(f"Average likes: {viral_videos['likeCount'].mean():,.0f}")
            print(f"Average comments: {viral_videos['commentCount'].mean():,.0f}")
            
            # Common words in viral titles
            all_viral_words = []
            for title in viral_videos['title']:
                if pd.notna(title):
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(title).lower())
                    all_viral_words.extend(words)
            
            viral_word_counts = Counter(all_viral_words)
            print(f"\n--- Top 15 Words in Viral Titles ---")
            for word, count in viral_word_counts.most_common(15):
                print(f"{word}: {count}")
            
            return viral_videos[['title', 'viewCount', 'likeCount', 'commentCount', 'engagement_rate']]
        
        return "No viral videos found in this period"

# Example usage function
def run_trend_analysis(videos_df):
    """Run comprehensive trend analysis"""
    
    print("🔥 TREND SPOTTING ANALYSIS 🔥")
    print("=" * 50)
    
    # Initialize trend spotter
    spotter = TrendSpotter(videos_df)
    
    # 1. Find trending keywords in titles (last 7 days)
    print("\n📈 TRENDING KEYWORDS IN TITLES (Last 7 days)")
    trending_titles = spotter.find_trending_keywords(source='title', time_period='Last_7_days')
    if not trending_titles.empty:
        print(trending_titles.head(10)[['keyword', 'frequency', 'avg_views', 'avg_engagement']].to_string())
    else:
        print("No trending keywords found")
    
    # 2. Find trending tags
    print("\n🏷️ TRENDING TAGS (Last 7 days)")
    trending_tags = spotter.find_trending_keywords(source='tag', time_period='Last_7_days')
    if not trending_tags.empty:
        print(trending_tags.head(10)[['keyword', 'frequency', 'avg_views', 'avg_engagement']].to_string())
    else:
        print("No trending tags found")
    
    # 3. Detect viral patterns
    print("\n💥 VIRAL CONTENT ANALYSIS")
    viral_analysis = spotter.detect_viral_patterns()
    
    # 4. Analyze momentum for specific keywords
    if not trending_titles.empty:
        top_keyword = trending_titles.iloc[0]['keyword']
        print(f"\n📊 MOMENTUM ANALYSIS for '{top_keyword}'")
        momentum = spotter.analyze_keyword_momentum(top_keyword)
        if momentum is not None:
            print(momentum)
    
    return spotter

# Load and run analysis
try:
    print("Loading videos.csv...")
    sys.stdout.reconfigure(encoding="utf-8")
    dataset = "dataset/dataset.zip"

    with ZipFile(dataset) as z:
        with z.open('videos.csv') as f:
            videos_df = pd.read_csv(f)
    
    print(f"Dataset loaded with shape: {videos_df.shape}")
    print("Available columns:")
    for i, col in enumerate(videos_df.columns):
        print(f"{i+1}. {col}")
    
    # Create column mapping - handle different naming conventions
    column_mapping = {}
    
    # Map common column variations
    for col in videos_df.columns:
        col_lower = col.lower()
        if 'view' in col_lower and 'count' in col_lower:
            column_mapping['viewCount'] = col
        elif 'like' in col_lower and 'count' in col_lower:
            column_mapping['likeCount'] = col
        elif 'comment' in col_lower and 'count' in col_lower:
            column_mapping['commentCount'] = col
        elif 'title' in col_lower:
            column_mapping['title'] = col
        elif 'description' in col_lower:
            column_mapping['description'] = col
        elif 'tag' in col_lower:
            column_mapping['tags'] = col
        elif 'published' in col_lower:
            column_mapping['publishedAt'] = col
        elif 'topic' in col_lower and 'categor' in col_lower:
            column_mapping['topicCategories'] = col
    
    print(f"\nColumn mapping found: {column_mapping}")
    
    # Rename columns to standard names
    videos_df = videos_df.rename(columns=column_mapping)
    
    # Fill missing values for available columns
    if 'viewCount' in videos_df.columns:
        videos_df['viewCount'] = videos_df['viewCount'].fillna(0)
    else:
        videos_df['viewCount'] = 0  # Create dummy column
        
    if 'likeCount' in videos_df.columns:
        videos_df['likeCount'] = videos_df['likeCount'].fillna(0)
    else:
        videos_df['likeCount'] = 0  # Create dummy column
        
    if 'commentCount' in videos_df.columns:
        videos_df['commentCount'] = videos_df['commentCount'].fillna(0)
    else:
        videos_df['commentCount'] = 0  # Create dummy column
    
    if 'title' in videos_df.columns:
        videos_df['title'] = videos_df['title'].fillna('')
    else:
        print("Warning: No title column found!")
        videos_df['title'] = ''
        
    if 'description' in videos_df.columns:
        videos_df['description'] = videos_df['description'].fillna('')
    else:
        videos_df['description'] = ''
        
    if 'tags' in videos_df.columns:
        videos_df['tags'] = videos_df['tags'].fillna('')
    else:
        videos_df['tags'] = ''
    
    # Check if we have publishedAt column
    if 'publishedAt' not in videos_df.columns:
        print("Warning: No publishedAt column found! Using dummy dates.")
        # Create dummy dates for demonstration
        videos_df['publishedAt'] = pd.date_range(end='2025-01-01', periods=len(videos_df), freq='1D')
    
    print(f"\nProcessed dataset shape: {videos_df.shape}")
    print("Final columns:", list(videos_df.columns))
    
    # Run trend analysis
    trend_spotter = run_trend_analysis(videos_df)
    
    print("\n[SUCCESS] Trend analysis complete!")
    print("\nYou can now use:")
    print("- trend_spotter.find_trending_keywords() for specific time periods")
    print("- trend_spotter.analyze_keyword_momentum() for keyword tracking")
    print("- trend_spotter.detect_viral_patterns() for viral content analysis")
    
except FileNotFoundError:
    print("[ERROR] videos.csv not found. Please make sure the file exists in the dataset/ folder")
except UnicodeEncodeError:
    print("[ERROR] Console encoding issue. Try running in a different terminal.")
except Exception as e:
    print(f"[ERROR] {str(e)}")
    print("Please check your videos.csv file structure and try again.")