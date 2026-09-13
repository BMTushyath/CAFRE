import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_biased_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Non-sensitive features
    experience_years = np.random.uniform(0, 20, n_samples)
    education_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]) # 1: High School, 2: Bachelors, 3: Masters, 4: PhD
    
    # 2. Sensitive attribute: Gender (M/F) with some representation bias
    # Let's say Males are 70% and Females are 30% in this domain
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3])
    
    # Another sensitive attribute: Race (A/B/C)
    race = np.random.choice(['GroupA', 'GroupB', 'GroupC'], n_samples, p=[0.6, 0.25, 0.15])
    
    # 3. Target variable: 'income' (binary: >50K or <=50K)
    # We introduce BIAS:
    # Male gender adds to the score, GroupA adds to the score.
    # While experience and education also add to the score.
    
    score = (
        experience_years * 1.5 + 
        education_level * 5.0 +
        np.where(gender == 'Male', 10.0, -5.0) + # Bias against Female
        np.where(race == 'GroupA', 5.0, np.where(race == 'GroupB', -2.0, -8.0)) # Bias against GroupB/C
    )
    
    # Add some noise
    score += np.random.normal(0, 5, n_samples)
    
    # Threshold for income > 50K
    threshold = np.median(score)
    income = np.where(score > threshold, '>50K', '<=50K')
    
    df = pd.DataFrame({
        'experience_years': np.round(experience_years, 1),
        'education_level': education_level,
        'gender': gender,
        'race': race,
        'income': income
    })
    
    return df

if __name__ == "__main__":
    df = create_biased_dataset(n_samples=1500)
    
    # Split into training and testing datasets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv("biased_dataset_train.csv", index=False)
    test_df.to_csv("biased_dataset_test.csv", index=False)
    
    print("Created biased_dataset_train.csv (1200 rows) and biased_dataset_test.csv (300 rows)")
    print("\nGender distribution:")
    print(df['gender'].value_counts(normalize=True))
    print("\nRace distribution:")
    print(df['race'].value_counts(normalize=True))
    print("\nIncome >50K by Gender:")
    print(df[df['income'] == '>50K']['gender'].value_counts(normalize=True))
