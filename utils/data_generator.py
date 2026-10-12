"""
utils/data_generator.py
------------------------
Generates a realistic, MESSY synthetic retail sales CSV so the pipeline
has something interesting to profile, clean, and aggregate.

Intentional mess injected:
- Mixed date formats (ISO, US, "5-Jan-2024")
- Inconsistent store name casing / whitespace
- Currency symbols and commas in unit_price
- Negative quantities (returns) and a few unparsable quantities ("N/A")
- Missing customer_id values
- A handful of exact duplicate rows
"""

import random
from datetime import datetime, timedelta

import pandas as pd

try:
    from faker import Faker
    fake = Faker()
    Faker.seed(42)
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False

random.seed(42)

STORES = [
    "Downtown Flagship", "  downtown flagship ", "Uptown Mall", "UPTOWN MALL",
    "Riverside Outlet", "Riverside outlet", "Airport Kiosk", "Suburban Center",
]
CATEGORIES = ["Apparel", "Electronics", "Home & Kitchen", "Footwear", "Accessories", "Beauty"]
PRODUCTS = {
    "Apparel": ["Denim Jacket", "Cotton Tee", "Wool Sweater", "Chino Pants"],
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Phone Charger", "Smart Watch"],
    "Home & Kitchen": ["Coffee Maker", "Non-stick Pan", "Blender", "Cutlery Set"],
    "Footwear": ["Running Shoes", "Leather Boots", "Sandals", "Sneakers"],
    "Accessories": ["Leather Wallet", "Sunglasses", "Baseball Cap", "Tote Bag"],
    "Beauty": ["Moisturizer", "Lipstick Set", "Shampoo", "Perfume"],
}
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Cash", "Gift Card", "Mobile Wallet"]
REGIONS = ["North", "South", "East", "West"]

DATE_FORMATS = [
    lambda d: d.strftime("%Y-%m-%d"),
    lambda d: d.strftime("%m/%d/%Y"),
    lambda d: f"{d.day}-{d.strftime('%b')}-{d.year}",
]


def _random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def _messy_price(base_price: float) -> str:
    roll = random.random()
    if roll < 0.35:
        return f"${base_price:,.2f}"
    elif roll < 0.45:
        return f"{base_price:,.2f} USD"
    elif roll < 0.5:
        return ""
    return f"{base_price:.2f}"


def generate_raw_sales_csv(output_path, n_rows: int = 600) -> None:
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)

    rows = []
    for i in range(1, n_rows + 1):
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        base_price = round(random.uniform(8, 250), 2)
        order_date_dt = _random_date(start_date, end_date)
        date_fmt = random.choice(DATE_FORMATS)

        quantity = random.randint(1, 5)
        if random.random() < 0.06:
            quantity = -abs(quantity)
        if random.random() < 0.02:
            quantity = "N/A"

        customer_id = f"CUST{1000 + random.randint(1, 250)}"
        if random.random() < 0.05:
            customer_id = ""

        row = {
            "order_id": f"ORD{100000 + i}",
            "order_date": date_fmt(order_date_dt),
            "store_id": f"ST{random.randint(1, 8):02d}",
            "store_name": random.choice(STORES),
            "product_id": f"P{random.randint(1000, 1050)}",
            "product_category": category,
            "product_name": product,
            "quantity": quantity,
            "unit_price": _messy_price(base_price),
            "discount_pct": round(random.choice([0, 0, 0, 5, 10, 15, 20]), 2),
            "customer_id": customer_id,
            "payment_method": random.choice(PAYMENT_METHODS),
            "region": random.choice(REGIONS),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    dup_sample = df.sample(n=15, random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"[DATA_GENERATOR] Generated {len(df)} rows (incl. duplicates) → {output_path}")
