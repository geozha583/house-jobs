import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# --- One-time setup for NLTK ---
# You may need to run this command once to download the stop words list
# nltk.download('stopwords')

def get_congress_number(year):
    """
    Calculates the Congress number for a given year.
    Each Congress lasts for two years.
    """
    if year < 1789:
        return None
    return (year - 1789) // 2 + 1

def clean_text_nltk(text):
    """
    Removes stop words (using NLTK's list) and punctuation from a given text.
    """
    if not isinstance(text, str):
        return ""
    # Use the English stop words list from NLTK
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuation and convert to lower case
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Filter out stop words
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Load the datasets
job_listings_df = pd.read_csv("enriched_job_listings.csv")
legislators_df = pd.read_csv("https://unitedstates.github.io/congress-legislators/legislators-current.csv")

# Create a dummy 'Year' column and correct the 'Congress_Number'
job_listings_df['Year'] = job_listings_df.index.map(lambda x: 2023 if x % 2 == 0 else 2024)
job_listings_df['Congress_Number'] = job_listings_df['Year'].apply(get_congress_number)

# Separate into members and committees
committees_df = job_listings_df[job_listings_df['Posting_Author'].str.contains("committee", case=False, na=False)].copy()
members_df = job_listings_df[~job_listings_df['Posting_Author'].str.contains("committee", case=False, na=False)].copy()

# Add gender information
legislators_df['full_name'] = legislators_df['first_name'] + ' ' + legislators_df['last_name']
members_df['Posting_Author_Lower'] = members_df['Posting_Author'].str.lower()
legislators_df['full_name_lower'] = legislators_df['full_name'].str.lower()

merged_df = pd.merge(members_df, legislators_df[['full_name_lower', 'gender']], left_on='Posting_Author_Lower', right_on='full_name_lower', how='left')
merged_df['Gender'] = merged_df['gender'].apply(lambda x: 1 if x == 'F' else 0 if x == 'M' else None)
members_df = merged_df.drop(columns=['Posting_Author_Lower', 'full_name_lower', 'gender'])

# Remove stop words using the new NLTK function
members_df['Posting_Text'] = members_df['Posting_Text'].apply(clean_text_nltk)
committees_df['Posting_Text'] = committees_df['Posting_Text'].apply(clean_text_nltk)

# Display the cleaned dataframes
print("### Individual Members of Congress Job Postings")
print(members_df.head().to_markdown(index=False))
print("\n### Committee Job Postings")
print(committees_df.head().to_markdown(index=False))

# Provide download links
members_df.to_csv("members_job_postings_nltk.csv", index=False)
committees_df.to_csv("committees_job_postings_nltk.csv", index=False)
