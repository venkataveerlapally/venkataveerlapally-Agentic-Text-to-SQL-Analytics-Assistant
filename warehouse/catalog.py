"""Schema catalog for the 12-table retail warehouse.

This is the single source of truth for table/column names, types, descriptions
and foreign keys. It is used to:
  * build the local SQLite warehouse (warehouse/seed.py)
  * create the Snowflake tables (warehouse/load_snowflake.py)
  * enrich `describe_table` results in the MCP server with business descriptions
"""

from __future__ import annotations

# type tokens: INT, TEXT, REAL, DATE, BOOL
# Each column: (name, type, description, foreign_key or None)
CATALOG: dict[str, dict] = {
    "regions": {
        "description": "Sales regions. Every store and customer belongs to one region.",
        "columns": [
            ("region_id", "INT", "Primary key", None),
            ("region_name", "TEXT", "Region name: North, South, East, West, Central", None),
        ],
    },
    "stores": {
        "description": "Physical stores plus one 'Online' store used for web orders.",
        "columns": [
            ("store_id", "INT", "Primary key", None),
            ("store_name", "TEXT", "Store display name", None),
            ("region_id", "INT", "Region the store is in", "regions.region_id"),
            ("city", "TEXT", "City", None),
            ("opened_date", "DATE", "Date the store opened", None),
        ],
    },
    "customers": {
        "description": "Registered customers.",
        "columns": [
            ("customer_id", "INT", "Primary key", None),
            ("first_name", "TEXT", "First name", None),
            ("last_name", "TEXT", "Last name", None),
            ("email", "TEXT", "Email address", None),
            ("region_id", "INT", "Home region of the customer", "regions.region_id"),
            ("signup_date", "DATE", "Date the customer registered", None),
            ("segment", "TEXT", "Customer segment: Consumer, Small Business, Enterprise", None),
        ],
    },
    "categories": {
        "description": "Product categories.",
        "columns": [
            ("category_id", "INT", "Primary key", None),
            ("category_name", "TEXT", "Category name, e.g. Electronics, Home, Toys", None),
        ],
    },
    "suppliers": {
        "description": "Suppliers that provide products.",
        "columns": [
            ("supplier_id", "INT", "Primary key", None),
            ("supplier_name", "TEXT", "Supplier company name", None),
            ("country", "TEXT", "Supplier country", None),
        ],
    },
    "products": {
        "description": "Product catalog with list price and unit cost.",
        "columns": [
            ("product_id", "INT", "Primary key", None),
            ("product_name", "TEXT", "Product name", None),
            ("category_id", "INT", "Product category", "categories.category_id"),
            ("supplier_id", "INT", "Supplier of the product", "suppliers.supplier_id"),
            ("list_price", "REAL", "Current list price in USD", None),
            ("unit_cost", "REAL", "Cost to the company per unit in USD", None),
            ("is_active", "BOOL", "1 if the product is still sold, else 0", None),
        ],
    },
    "employees": {
        "description": "Store employees. manager_id points to another employee.",
        "columns": [
            ("employee_id", "INT", "Primary key", None),
            ("first_name", "TEXT", "First name", None),
            ("last_name", "TEXT", "Last name", None),
            ("store_id", "INT", "Store the employee works at", "stores.store_id"),
            ("role", "TEXT", "Role: Store Manager, Sales Associate, Cashier", None),
            ("hire_date", "DATE", "Hire date", None),
            ("manager_id", "INT", "Employee's manager (NULL for store managers)", "employees.employee_id"),
        ],
    },
    "orders": {
        "description": (
            "Order headers. Revenue is NOT stored here: compute it from order_items as "
            "SUM(quantity * unit_price * (1 - discount)). Exclude status = 'Cancelled' "
            "for revenue unless asked otherwise."
        ),
        "columns": [
            ("order_id", "INT", "Primary key", None),
            ("customer_id", "INT", "Customer who placed the order", "customers.customer_id"),
            ("store_id", "INT", "Store that fulfilled the order", "stores.store_id"),
            ("employee_id", "INT", "Employee who handled the order (NULL for online)", "employees.employee_id"),
            ("order_date", "DATE", "Date the order was placed", None),
            ("status", "TEXT", "Order status: Completed, Shipped, Pending, Cancelled", None),
            ("channel", "TEXT", "Sales channel: In-Store or Online", None),
        ],
    },
    "order_items": {
        "description": "Order line items. Line revenue = quantity * unit_price * (1 - discount).",
        "columns": [
            ("order_item_id", "INT", "Primary key", None),
            ("order_id", "INT", "Parent order", "orders.order_id"),
            ("product_id", "INT", "Product sold", "products.product_id"),
            ("quantity", "INT", "Units sold", None),
            ("unit_price", "REAL", "Price per unit at time of sale in USD", None),
            ("discount", "REAL", "Discount fraction between 0 and 0.3", None),
        ],
    },
    "payments": {
        "description": "Payments received for orders. An order can have one or more payments.",
        "columns": [
            ("payment_id", "INT", "Primary key", None),
            ("order_id", "INT", "Order paid for", "orders.order_id"),
            ("payment_date", "DATE", "Date of payment", None),
            ("amount", "REAL", "Amount paid in USD", None),
            ("method", "TEXT", "Payment method: Credit Card, Debit Card, PayPal, Gift Card", None),
        ],
    },
    "shipments": {
        "description": "Shipments for online orders. delivered_date is NULL if not yet delivered.",
        "columns": [
            ("shipment_id", "INT", "Primary key", None),
            ("order_id", "INT", "Order shipped", "orders.order_id"),
            ("carrier", "TEXT", "Carrier: UPS, FedEx, USPS, DHL", None),
            ("shipped_date", "DATE", "Date shipped", None),
            ("delivered_date", "DATE", "Date delivered (NULL if in transit)", None),
        ],
    },
    "returns": {
        "description": "Returned order line items.",
        "columns": [
            ("return_id", "INT", "Primary key", None),
            ("order_item_id", "INT", "Returned line item", "order_items.order_item_id"),
            ("return_date", "DATE", "Date of the return", None),
            ("reason", "TEXT", "Reason: Damaged, Wrong Item, Not as Described, Changed Mind", None),
            ("refund_amount", "REAL", "Refund issued in USD", None),
        ],
    },
}

TABLE_ORDER = list(CATALOG.keys())  # dependency-safe insert order

SQLITE_TYPES = {"INT": "INTEGER", "TEXT": "TEXT", "REAL": "REAL", "DATE": "TEXT", "BOOL": "INTEGER"}
SNOWFLAKE_TYPES = {"INT": "NUMBER(38,0)", "TEXT": "VARCHAR", "REAL": "NUMBER(12,2)", "DATE": "DATE", "BOOL": "NUMBER(1,0)"}


def column_names(table: str) -> list[str]:
    return [c[0] for c in CATALOG[table]["columns"]]


def ddl(table: str, dialect: str) -> str:
    types = SQLITE_TYPES if dialect == "sqlite" else SNOWFLAKE_TYPES
    cols = []
    for i, (name, typ, _desc, _fk) in enumerate(CATALOG[table]["columns"]):
        line = f"{name} {types[typ]}"
        if i == 0:
            line += " PRIMARY KEY"
        cols.append(line)
    for name, _typ, _desc, fk in CATALOG[table]["columns"]:
        if fk:
            ref_table, ref_col = fk.split(".")
            cols.append(f"FOREIGN KEY ({name}) REFERENCES {ref_table}({ref_col})")
    return f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols) + "\n)"
