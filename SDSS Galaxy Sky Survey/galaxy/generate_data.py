
import pandas as pd
import numpy as np

def generate_large_dataset(n_samples=5000):
    np.random.seed(42)
    
    # Generate random features
    # u, g, r, i, z mostly correlate but have some variance
    # Redshift correlates with 'far away' galaxies but let's just make it random-ish for this classification task
    
    data = {
        'u': np.random.normal(18, 2, n_samples),
        'g': np.random.normal(17, 2, n_samples),
        'r': np.random.normal(16, 2, n_samples),
        'i': np.random.normal(15, 2, n_samples),
        'z': np.random.normal(14, 2, n_samples),
        'redshift': np.random.exponential(0.1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on some logic so the model has something to learn
    # Let's say STARBURST galaxies have higher UV (u-band) and higher redshift on average
    
    # Simple logic: if (u - r) > 2.0 or redshift > 0.15 => STARBURST, else STARFORMING
    # Adding some noise
    
    conditions = [
        (df['u'] - df['r'] > 1.5) & (df['redshift'] > 0.1),
        (df['redshift'] > 0.25)
    ]
    choices = ['STARBURST', 'STARBURST']
    
    df['subclass'] = np.select(conditions, choices, default='STARFORMING')
    
    # Add some random noise to flip labels to make it realistic
    mask = np.random.random(n_samples) < 0.1 # 10% noise
    df.loc[mask, 'subclass'] = np.where(df.loc[mask, 'subclass'] == 'STARBURST', 'STARFORMING', 'STARBURST')

    print(f"Generated {n_samples} samples.")
    print(df['subclass'].value_counts())
    
    df.to_csv('galaxy_data.csv', index=False)
    print("Saved to galaxy_data.csv")

if __name__ == "__main__":
    generate_large_dataset()
