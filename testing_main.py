import pandas as pd
import re
from print_columns import show_data_in_window

# Load the dataset
df = pd.read_csv('videos.csv')
print(df.head())
print(df.columns)

# Filter comments with likeCount above average
def filter_likeCount_above_average(df):
    average_like_count = df['likeCount'].mean()
    comments_above_average = df[(df['likeCount'] >= average_like_count)]
    # Sort by likeCount in descending order for the top comments
    sorted_like_counts = comments_above_average.sort_values(by='likeCount', ascending=False)
    return sorted_like_counts

# Remove rows where likeCount is NaN
def filter_likeCount_non_zero(df):
    filtered_df = df.dropna(subset=['likeCount'])
    return filtered_df

# Remove rows with empty tags
def filter_data_empty_tags(df):
    filtered_df = df.dropna(subset=['tags'])
    return filtered_df

# Remove rows with empty descriptions
def filter_data_empty_description(df):
    filtered_df = df.dropna(subset=['description'])
    return filtered_df

# Search for a specific word in the 'description' column
def appearance_specific_word(df, word):
    word = word.lower()  # Convert to lowercase for case-insensitive search
    # Use .loc to modify the 'contains_word' column properly
    df.loc[:, 'contains_word'] = df['description'].str.contains(r'\b' + re.escape(word) + r'\b', case=False, na=False)
    # Display rows where the word appears
    word_appearances = df[df['contains_word']]
    return word_appearances

def look_for_ratio(molecular, denomimator):
    return molecular/denomimator

# Select relevant columns for display
columns_to_display = ['title', 'tags', 'description', 'likeCount', 'viewCount', 'topicCategories']

# Apply the filtering and checking steps
sorted_likeCount = filter_likeCount_above_average(df)
filtered_df = sorted_likeCount[columns_to_display]
non_empty_description = filter_data_empty_description(filtered_df)
print(len(non_empty_description))

# Check for a specific word, e.g., 'mask'
check_mask = appearance_specific_word(non_empty_description, 'mask')
print(len(check_mask))
ratio_for_appearance_mask = look_for_ratio(len(check_mask), len(non_empty_description))
print(ratio_for_appearance_mask)

check_face = appearance_specific_word(non_empty_description, 'face')
print(len(check_face))
ratio_for_appearance_face = look_for_ratio(len(check_face), len(non_empty_description))
print(ratio_for_appearance_face)

# Show the data in the window
show_data_in_window(check_mask)
