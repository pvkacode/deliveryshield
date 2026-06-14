import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_delivery_dataset(n_samples=50000):
    """
    Generates a realistic delivery dataset inspired by the Amazon ALMRRC dataset.
    Features mirror real operational variables used in last-mile delivery research.
    """

    # --- Time features ---
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(
        days=np.random.randint(0, 730),
        hours=np.random.randint(8, 22),
        minutes=np.random.randint(0, 59)
    ) for _ in range(n_samples)]

    hour_of_day = np.array([t.hour for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])
    month = np.array([t.month for t in timestamps])
    is_weekend = (day_of_week >= 5).astype(int)

    # --- Address features ---
    address_types = np.random.choice(
        ['house', 'apartment', 'gated_community', 'business', 'po_box'],
        size=n_samples,
        p=[0.35, 0.30, 0.15, 0.15, 0.05]
    )

    # Apartments and gated communities fail more
    address_type_encoded = pd.Categorical(address_types).codes

    # --- Package features ---
    package_sizes = np.random.choice(
        ['small', 'medium', 'large', 'extra_large'],
        size=n_samples,
        p=[0.40, 0.35, 0.18, 0.07]
    )
    package_size_encoded = pd.Categorical(package_sizes).codes
    package_weight_kg = np.random.exponential(scale=2.5, size=n_samples).clip(0.1, 30)

    # --- Customer / instruction features ---
    has_delivery_note = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    note_quality = np.where(
        has_delivery_note == 1,
        np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.5, 0.3]),
        0
    )
    customer_home_probability = np.random.beta(a=3, b=2, size=n_samples)

    # --- Location features ---
    zip_failure_rate = np.random.beta(a=2, b=8, size=n_samples)  # Most zips have low failure rate
    distance_from_depot_km = np.random.exponential(scale=12, size=n_samples).clip(0.5, 80)
    urban_density = np.random.choice(
        ['urban', 'suburban', 'rural'],
        size=n_samples,
        p=[0.50, 0.35, 0.15]
    )
    urban_density_encoded = pd.Categorical(urban_density).codes

    # --- Driver features ---
    driver_experience_months = np.random.choice(
        np.arange(1, 61),
        size=n_samples
    )
    driver_daily_deliveries = np.random.randint(30, 120, size=n_samples)
    driver_fatigue_score = (driver_daily_deliveries / 120).clip(0, 1)  # 0=fresh, 1=exhausted

    # --- Weather features ---
    weather_conditions = np.random.choice(
        ['clear', 'cloudy', 'rain', 'heavy_rain', 'snow'],
        size=n_samples,
        p=[0.45, 0.25, 0.18, 0.07, 0.05]
    )
    weather_encoded = pd.Categorical(weather_conditions).codes
    temperature_celsius = np.random.normal(loc=18, scale=10, size=n_samples).clip(-10, 45)

    # --- Prior attempt features ---
    prior_failed_attempts = np.random.choice(
        [0, 1, 2, 3],
        size=n_samples,
        p=[0.75, 0.15, 0.07, 0.03]
    )

    # --- Peak hour ---
    is_peak_hour = ((hour_of_day >= 17) & (hour_of_day <= 20)).astype(int)

    # --- Construct failure probability (ground truth simulation) ---
    failure_logit = (
        -3.5
        + 0.8  * (address_types == 'apartment').astype(int)
        + 1.2  * (address_types == 'gated_community').astype(int)
        + 0.5  * (address_types == 'po_box').astype(int)
        - 0.6  * has_delivery_note
        - 0.4  * (note_quality == 3).astype(int)
        + 2.5  * zip_failure_rate
        + 0.4  * (weather_conditions == 'heavy_rain').astype(int)
        + 0.3  * (weather_conditions == 'rain').astype(int)
        + 0.6  * (weather_conditions == 'snow').astype(int)
        + 0.5  * is_peak_hour
        + 0.3  * is_weekend
        + 0.8  * prior_failed_attempts
        + 0.3  * driver_fatigue_score
        - 0.5  * (driver_experience_months > 24).astype(int)
        - 0.4  * customer_home_probability
        + 0.3  * (package_sizes == 'extra_large').astype(int)
        + 0.2  * (urban_density == 'rural').astype(int)
        + np.random.normal(0, 0.3, n_samples)  # noise
    )

    failure_prob = 1 / (1 + np.exp(-failure_logit))
    failed = (np.random.rand(n_samples) < failure_prob).astype(int)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'address_type': address_types,
        'address_type_encoded': address_type_encoded,
        'package_size': package_sizes,
        'package_size_encoded': package_size_encoded,
        'package_weight_kg': package_weight_kg.round(2),
        'has_delivery_note': has_delivery_note,
        'note_quality': note_quality,
        'customer_home_probability': customer_home_probability.round(3),
        'zip_failure_rate': zip_failure_rate.round(3),
        'distance_from_depot_km': distance_from_depot_km.round(2),
        'urban_density': urban_density,
        'urban_density_encoded': urban_density_encoded,
        'driver_experience_months': driver_experience_months,
        'driver_daily_deliveries': driver_daily_deliveries,
        'driver_fatigue_score': driver_fatigue_score.round(3),
        'weather_condition': weather_conditions,
        'weather_encoded': weather_encoded,
        'temperature_celsius': temperature_celsius.round(1),
        'prior_failed_attempts': prior_failed_attempts,
        'delivery_failed': failed
    })

    return df


if __name__ == '__main__':
    print("Generating delivery dataset...")
    df = generate_delivery_dataset(50000)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/deliveries.csv', index=False)

    print(f"Dataset saved: {len(df)} rows")
    print(f"Failure rate: {df['delivery_failed'].mean():.1%}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))
