"""Generate a deterministic synthetic retail dataset and build the SQLite warehouse.

Usage:
    python -m warehouse.seed              # writes data/warehouse.db
    python -m warehouse.seed --out x.db

The same generated rows are used by warehouse/load_snowflake.py, so the local
and Snowflake warehouses contain identical data (handy for evaluation).
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from warehouse.catalog import CATALOG, TABLE_ORDER, column_names, ddl

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"
START, END = date(2024, 1, 1), date(2025, 12, 31)

FIRST = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "David",
         "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas",
         "Sarah", "Priya", "Wei", "Carlos", "Aisha", "Hiro", "Fatima", "Ivan", "Sofia", "Omar", "Ana"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
        "Martinez", "Patel", "Kim", "Nguyen", "Chen", "Singh", "Lopez", "Wilson", "Anderson",
        "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Khan", "Silva"]
REGIONS = {
    "North": ["Minneapolis", "Chicago", "Detroit"],
    "South": ["Atlanta", "Houston", "Miami"],
    "East": ["New York", "Boston", "Philadelphia"],
    "West": ["Seattle", "San Francisco", "Los Angeles"],
    "Central": ["Denver", "Kansas City", "St. Louis"],
}
CATEGORIES = {
    "Electronics": (["Headphones", "Smartwatch", "Bluetooth Speaker", "Tablet", "Webcam", "Charger"], (20, 600)),
    "Home": (["Lamp", "Throw Blanket", "Cookware Set", "Coffee Maker", "Vacuum", "Air Purifier"], (15, 350)),
    "Toys": (["Puzzle", "Building Blocks", "Board Game", "RC Car", "Plush Bear", "Art Kit"], (8, 120)),
    "Sports": (["Yoga Mat", "Dumbbells", "Running Shoes", "Tennis Racket", "Water Bottle", "Bike Helmet"], (10, 250)),
    "Beauty": (["Moisturizer", "Shampoo", "Perfume", "Hair Dryer", "Lip Balm", "Face Serum"], (5, 150)),
    "Books": (["Cookbook", "Novel", "Travel Guide", "Biography", "Kids Picture Book", "Notebook"], (6, 60)),
    "Grocery": (["Coffee Beans", "Olive Oil", "Tea Sampler", "Dark Chocolate", "Granola", "Hot Sauce"], (4, 40)),
    "Office": (["Desk Chair", "Monitor Stand", "Pen Set", "Desk Organizer", "Planner", "Keyboard"], (5, 400)),
}
ADJ = ["Classic", "Pro", "Eco", "Deluxe", "Mini", "Ultra", "Smart", "Essential"]
COUNTRIES = ["USA", "China", "Germany", "Mexico", "Vietnam", "India", "Japan", "Canada"]
CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
METHODS = ["Credit Card", "Debit Card", "PayPal", "Gift Card"]
REASONS = ["Damaged", "Wrong Item", "Not as Described", "Changed Mind"]


def _d(d: date) -> str:
    return d.isoformat()


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate(seed: int = 42) -> dict[str, list[tuple]]:
    rng = random.Random(seed)
    data: dict[str, list[tuple]] = {t: [] for t in TABLE_ORDER}

    # regions
    region_ids = {}
    for i, name in enumerate(REGIONS, start=1):
        data["regions"].append((i, name))
        region_ids[name] = i

    # stores: 3 per region + 1 online store
    sid = 0
    for rname, cities in REGIONS.items():
        for city in cities:
            sid += 1
            data["stores"].append((sid, f"{city} Store", region_ids[rname], city,
                                   _d(_rand_date(rng, date(2015, 1, 1), date(2023, 6, 30)))))
    online_store_id = sid + 1
    data["stores"].append((online_store_id, "Online", region_ids["Central"], "Online", "2018-03-01"))
    physical_stores = [s[0] for s in data["stores"] if s[0] != online_store_id]

    # customers
    segments = ["Consumer"] * 7 + ["Small Business"] * 2 + ["Enterprise"]
    for cid in range(1, 2001):
        fn, ln = rng.choice(FIRST), rng.choice(LAST)
        data["customers"].append((cid, fn, ln, f"{fn.lower()}.{ln.lower()}{cid}@example.com",
                                  rng.randint(1, 5), _d(_rand_date(rng, date(2021, 1, 1), END)),
                                  rng.choice(segments)))

    # categories, suppliers, products
    for i, cat in enumerate(CATEGORIES, start=1):
        data["categories"].append((i, cat))
    for i in range(1, 21):
        data["suppliers"].append((i, f"Supplier {chr(64 + ((i - 1) % 26) + 1)}{i:02d} Co.", rng.choice(COUNTRIES)))
    pid = 0
    for cat_id, (cat, (items, (lo, hi))) in enumerate(CATEGORIES.items(), start=1):
        for item in items:
            for adj in rng.sample(ADJ, 3):
                pid += 1
                price = round(rng.uniform(lo, hi), 2)
                cost = round(price * rng.uniform(0.35, 0.75), 2)
                data["products"].append((pid, f"{adj} {item}", cat_id, rng.randint(1, 20), price, cost,
                                         0 if rng.random() < 0.08 else 1))
    products = data["products"]

    # employees: 1 manager + 5-7 staff per physical store
    eid = 0
    store_staff: dict[int, list[int]] = {}
    for s in physical_stores:
        eid += 1
        mgr = eid
        data["employees"].append((mgr, rng.choice(FIRST), rng.choice(LAST), s, "Store Manager",
                                  _d(_rand_date(rng, date(2016, 1, 1), date(2022, 12, 31))), None))
        store_staff[s] = [mgr]
        for _ in range(rng.randint(5, 7)):
            eid += 1
            role = rng.choice(["Sales Associate", "Sales Associate", "Cashier"])
            data["employees"].append((eid, rng.choice(FIRST), rng.choice(LAST), s, role,
                                      _d(_rand_date(rng, date(2018, 1, 1), date(2025, 6, 30))), mgr))
            store_staff[s].append(eid)

    # orders and children
    oi_id = pay_id = ship_id = ret_id = 0
    region_store = {r: [s[0] for s in data["stores"] if s[2] == r and s[0] != online_store_id] for r in range(1, 6)}
    for oid in range(1, 12001):
        cust = data["customers"][rng.randint(0, 1999)]
        # seasonality: more orders in Nov/Dec, growth in 2025
        while True:
            od = _rand_date(rng, START, END)
            weight = 1.0 + (0.6 if od.month in (11, 12) else 0) + (0.15 if od.year == 2025 else 0)
            if rng.random() < weight / 1.75:
                break
        online = rng.random() < 0.4
        store = online_store_id if online else rng.choice(region_store[cust[4]])
        emp = None if online else rng.choice(store_staff[store])
        status = rng.choices(["Completed", "Shipped", "Pending", "Cancelled"], [80, 8, 4, 8])[0]
        if od < date(2025, 11, 1) and status in ("Shipped", "Pending"):
            status = "Completed"
        data["orders"].append((oid, cust[0], store, emp, _d(od), status, "Online" if online else "In-Store"))

        total = 0.0
        for prod in rng.sample(products, rng.choices([1, 2, 3, 4, 5], [35, 30, 18, 10, 7])[0]):
            oi_id += 1
            qty = rng.choices([1, 2, 3, 4], [60, 25, 10, 5])[0]
            unit_price = round(prod[4] * rng.uniform(0.95, 1.05), 2)
            disc = rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.3])
            data["order_items"].append((oi_id, oid, prod[0], qty, unit_price, disc))
            line = qty * unit_price * (1 - disc)
            total += line
            if status == "Completed" and rng.random() < 0.05:
                ret_id += 1
                data["returns"].append((ret_id, oi_id, _d(od + timedelta(days=rng.randint(3, 40))),
                                        rng.choice(REASONS), round(line * rng.choice([1.0, 1.0, 0.5]), 2)))
        total = round(total, 2)

        if status != "Cancelled":
            if rng.random() < 0.1 and total > 50:  # split payment
                first = round(total * 0.5, 2)
                for amt in (first, round(total - first, 2)):
                    pay_id += 1
                    data["payments"].append((pay_id, oid, _d(od), amt, rng.choice(METHODS)))
            else:
                pay_id += 1
                data["payments"].append((pay_id, oid, _d(od), total, rng.choice(METHODS)))

        if online and status in ("Completed", "Shipped"):
            ship_id += 1
            shipped = od + timedelta(days=rng.randint(0, 3))
            delivered = None if status == "Shipped" else _d(shipped + timedelta(days=rng.randint(1, 8)))
            data["shipments"].append((ship_id, oid, rng.choice(CARRIERS), _d(shipped), delivered))

    return data


def build_sqlite(path: Path = DEFAULT_DB, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    data = generate(seed)
    conn = sqlite3.connect(path)
    try:
        for table in TABLE_ORDER:
            conn.execute(ddl(table, "sqlite"))
            cols = column_names(table)
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                data[table],
            )
        conn.commit()
    finally:
        conn.close()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_DB))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    path = build_sqlite(Path(args.out), args.seed)
    conn = sqlite3.connect(path)
    print(f"Built {path}")
    for t in CATALOG:
        print(f"  {t:<12} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>7,} rows")
    conn.close()


if __name__ == "__main__":
    main()
