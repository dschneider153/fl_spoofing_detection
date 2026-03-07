import pandas as pd
import numpy as np

def synthesize_spoofing_labels(df):

    df = df.copy()

    # Strcutural features only to identify spoof-like events
    df['spoofing_score'] = 0.0
    df['spoofing_score'] += (df['relative_size'] > 3).astype(float) * 2
    lifetime_ms = np.expm1(df['log_lifetime'])  # convert back from log
    df['spoofing_score'] += (lifetime_ms < 1000).astype(float) * 2  # <1sec
    df['spoofing_score'] += (df['ended_with_cancel'] == 1).astype(float) * 1
    df['spoofing_score'] += ((df['distance_ticks'] > 5) & (df['distance_ticks'] < 50)).astype(float) * 2
    df['spoofing_score'] += ((df['distance_ticks'] > 10) & (df['distance_ticks'] < 50)).astype(float) * 1
    
    df['spoofing_score'] = df['spoofing_score'] / df['spoofing_score'].max()
    threshold = 0.8
    df['is_spoofing'] = (df['spoofing_score'] >= threshold).astype(int)

    return df

features_df = pd.read_csv('features/features_january.csv')
features_df = synthesize_spoofing_labels(features_df)
features_df.to_csv('data/Training and Testing/largedataset.csv')


print(f"Spoofing cases: {features_df['is_spoofing'].sum()}")
print(f"Legitimate cases: {(~features_df['is_spoofing'].astype(bool)).sum()}")
print(f"\nScore distribution:\n{features_df['spoofing_score'].describe()}")

# Obvious spoofing (score > 0.9)
print(features_df[features_df['spoofing_score'] > 0.9][
    ['relative_size', 'distance_ticks', 'log_lifetime', 
     'ended_with_cancel', 'midprice_change_1000ms']
].head(10))

# Obvious legitimate (score < 0.3)
print(features_df[features_df['spoofing_score'] < 0.3][
    ['relative_size', 'distance_ticks', 'log_lifetime', 
     'ended_with_cancel', 'midprice_change_1000ms']
].head(10))

print(features_df.groupby('is_spoofing')['signed_impact_after'].describe())
print(features_df.groupby('is_spoofing')['order_book_imbalance_shift'].describe())
print(features_df.groupby('is_spoofing')['spread_change_ticks'].describe())

print(abs(features_df['midprice_change_1000ms']).describe())
print((features_df['price_reversion']).describe())