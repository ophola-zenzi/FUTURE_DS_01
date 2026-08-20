from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# BASE_DIR is the directory containing this script. Using Path makes path operations OS-agnostic.
BASE_DIR = Path(__file__).resolve().parent
# Locations for the dataset and outputs (images, reports, dashboard)
DATA_PATH = BASE_DIR / "sales_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
# Ensure the outputs directory exists so saving images won't fail.
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = BASE_DIR / "sales_analysis_report.md"
DASHBOARD_PATH = BASE_DIR / "sales_dashboard.html"


def generate_sales_data() -> pd.DataFrame:
    """
    Generate a synthetic sales dataset and write it to DATA_PATH.

    The generated dataset includes random orders across regions, categories, and products.
    Each row includes OrderDate, Region, Category, Product, Quantity, UnitPrice, Discount,
    Revenue (after discount), and Profit (generated using a random profit margin).

    Returns the generated DataFrame.
    """
    # Fix the random seed for reproducible synthetic data
    np.random.seed(42)

    # Predefined regions and product categories to sample from
    regions = ["North", "South", "East", "West", "Central"]
    categories = {
        "Electronics": ["Laptop", "Phone", "Tablet", "Headphones"],
        "Furniture": ["Chair", "Desk", "Sofa", "Cabinet"],
        "Office Supplies": ["Notebook", "Pen Set", "Stapler", "Binder"],
    }

    rows = []

    # Create 240 synthetic orders spaced randomly across a single year (2024)
    for _ in range(240):
        # Random order date between 2024-01-01 and 2024-12-31
        order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="D")
        region = np.random.choice(regions)
        category = np.random.choice(list(categories.keys()))
        product = np.random.choice(categories[category])
        quantity = int(np.random.randint(1, 10))  # quantity per order (1..9)
        unit_price = round(np.random.uniform(20, 400), 2)  # unit price between $20 and $400
        discount = round(np.random.uniform(0.0, 0.15), 2)  # up to 15% discount

        # Revenue accounts for quantity, unit price and discount
        revenue = round(quantity * unit_price * (1 - discount), 2)
        # Profit margin sampled and applied to revenue to produce a synthetic profit value
        profit_margin = np.random.uniform(0.18, 0.35)
        profit = round(revenue * profit_margin, 2)

        rows.append(
            {
                "OrderDate": order_date,
                "Region": region,
                "Category": category,
                "Product": product,
                "Quantity": quantity,
                "UnitPrice": unit_price,
                "Discount": discount,
                "Revenue": revenue,
                "Profit": profit,
            }
        )

    # Build DataFrame, sort by date, save to CSV and return
    df = pd.DataFrame(rows)
    df = df.sort_values("OrderDate").reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_data() -> pd.DataFrame:
    """
    Load data from disk if it exists; otherwise generate synthetic data.

    This function ensures there is always a DataFrame to work with, by delegating
    to generate_sales_data() if the CSV file is missing.
    """
    if not DATA_PATH.exists():
        return generate_sales_data()
    df = pd.read_csv(DATA_PATH)
    # Convert OrderDate column to datetime so time-based operations work correctly
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the DataFrame:
    - Drop rows missing critical fields
    - Keep only positive Revenue and Profit records
    - Add a "Month" column for monthly aggregations (YYYY-MM format)

    Returns a cleaned copy of the DataFrame.
    """
    clean_df = df.copy()
    # Drop rows that are missing key columns needed for analysis
    clean_df = clean_df.dropna(subset=["OrderDate", "Region", "Category", "Product", "Revenue", "Profit"])
    # Keep only rows with positive revenue and profit (removes refunds or bad data)
    clean_df = clean_df[(clean_df["Revenue"] > 0) & (clean_df["Profit"] > 0)].copy()
    # Create a Month column (string) like "2024-03" for grouping
    clean_df["Month"] = clean_df["OrderDate"].dt.to_period("M").astype(str)
    return clean_df


def create_charts(df: pd.DataFrame) -> dict[str, str]:
    """
    Create and save a set of charts to the OUTPUT_DIR and return a mapping of chart names to file paths.

    Charts produced:
    - monthly revenue trend (line)
    - top revenue products (bar)
    - profit by region (bar)
    - revenue by category (pie)
    """
    # Aggregate monthly revenue and profit for the line chart
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(MonthlyRevenue=("Revenue", "sum"), MonthlyProfit=("Profit", "sum"))
    )
    # Convert Month string back to a datetime for plotting on x-axis
    monthly["MonthDate"] = pd.to_datetime(monthly["Month"]) 

    # Plot Monthly Revenue Trend
    plt.figure(figsize=(10, 4))
    plt.plot(monthly["MonthDate"], monthly["MonthlyRevenue"], marker="o", color="#1f77b4")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    monthly_path = OUTPUT_DIR / "monthly_revenue.png"
    plt.savefig(monthly_path, dpi=200)
    plt.close()

    # Top products by revenue (take top 8)
    product_revenue = (
        df.groupby("Product", as_index=False)[["Revenue"]]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(8)
    )
    plt.figure(figsize=(9, 4))
    plt.bar(product_revenue["Product"], product_revenue["Revenue"], color="#2ca02c")
    plt.title("Top Revenue Products")
    plt.xlabel("Product")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    products_path = OUTPUT_DIR / "top_products.png"
    plt.savefig(products_path, dpi=200)
    plt.close()

    # Profit by region (simple bar chart)
    region_profit = (
        df.groupby("Region", as_index=False)[["Profit"]]
        .sum()
        .sort_values("Profit", ascending=False)
    )
    plt.figure(figsize=(8, 4))
    plt.bar(region_profit["Region"], region_profit["Profit"], color="#ff7f0e")
    plt.title("Profit by Region")
    plt.xlabel("Region")
    plt.ylabel("Profit")
    plt.tight_layout()
    region_path = OUTPUT_DIR / "profit_by_region.png"
    plt.savefig(region_path, dpi=200)
    plt.close()

    # Revenue by category (pie chart)
    category_revenue = (
        df.groupby("Category", as_index=False)[["Revenue"]]
        .sum()
        .sort_values("Revenue", ascending=False)
    )
    plt.figure(figsize=(7, 4))
    plt.pie(category_revenue["Revenue"], labels=category_revenue["Category"].tolist(), autopct="%1.1f%%", startangle=90)
    plt.title("Revenue by Category")
    plt.tight_layout()
    category_path = OUTPUT_DIR / "revenue_by_category.png"
    plt.savefig(category_path, dpi=200)
    plt.close()

    # Return file paths as strings so other functions can reference them
    return {
        "monthly": str(monthly_path),
        "products": str(products_path),
        "region": str(region_path),
        "category": str(category_path),
    }


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Compute high-level summary metrics used in the report and dashboard.

    Metrics returned include total revenue, total profit, average order value,
    the top product/region/category, best month, and average monthly growth.
    """
    total_revenue = round(float(df["Revenue"].sum()), 2)
    total_profit = round(float(df["Profit"].sum()), 2)
    avg_order_value = round(float(df["Revenue"].mean()), 2)

    # Recompute monthly aggregates here to find best month and growth
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(MonthlyRevenue=("Revenue", "sum"), MonthlyProfit=("Profit", "sum"))
    )
    # Best month by revenue (idxmax gives the row index of the max monthly revenue)
    best_month = monthly.loc[monthly["MonthlyRevenue"].idxmax()]

    # Product, region and category aggregates for top performers
    product_revenue = (
        df.groupby("Product", as_index=False)[["Revenue"]]
        .sum()
        .sort_values("Revenue", ascending=False)
    )
    region_profit = (
        df.groupby("Region", as_index=False)[["Profit"]]
        .sum()
        .sort_values("Profit", ascending=False)
    )
    category_revenue = (
        df.groupby("Category", as_index=False)[["Revenue"]].sum().sort_values("Revenue", ascending=False))

    # Extract the top labels (first row after sorting)
    top_product = product_revenue.iloc[0]["Product"]
    top_region = region_profit.iloc[0]["Region"]
    top_category = category_revenue.iloc[0]["Category"]

    # Compute average month-over-month revenue percentage change, expressed as percent
    monthly_growth = monthly["MonthlyRevenue"].pct_change().dropna().mean() * 100

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "avg_order_value": avg_order_value,
        "top_product": top_product,
        "top_region": top_region,
        "top_category": top_category,
        "best_month": best_month["Month"],
        # Round to 1 decimal place for display in the report/dashboard
        "monthly_growth": round(float(monthly_growth), 1),
    }


def write_report(summary: dict[str, object], charts: dict[str, str]) -> None:
    """
    Create a Markdown report summarizing findings and embedding the generated charts.

    The function writes a markdown file at REPORT_PATH containing an executive summary,
    key metrics, insights, recommendations, and the images produced earlier.
    """
    # Convert chart file paths to filenames so the markdown references the outputs/ folder
    relative_charts = {key: Path(path).name for key, path in charts.items()}

    # A multi-line f-string constructs the report content using summary values
    report = f"""# Business Sales Performance Report

## Executive Summary
This sales analytics project highlights the strongest revenue drivers, profitable regions, and the most promising growth opportunities for a business team. The analysis uses a realistic sample sales dataset and is designed to be client-ready for internal presentations or portfolio submissions.

## Key Metrics
- Total Revenue: ${summary['total_revenue']:.2f}
- Total Profit: ${summary['total_profit']:.2f}
- Average Order Value: ${summary['avg_order_value']:.2f}
- Best Month: {summary['best_month']}
- Monthly Revenue Growth: {summary['monthly_growth']:.1f}%

## Key Insights
- The highest revenue product is **{summary['top_product']}**.
- The most profitable region is **{summary['top_region']}**.
- The strongest revenue category is **{summary['top_category']}**.
- Revenue growth is trending positively, suggesting that the business should focus on scaling the top-performing products and regions.

## Recommendations
1. Increase stock and marketing attention for **{summary['top_product']}** because it is the biggest revenue driver.
2. Expand sales efforts in **{summary['top_region']}** by increasing promotions, partnerships, and customer outreach.
3. Prioritize the **{summary['top_category']}** category for cross-sell and upsell campaigns.
4. Review pricing and campaign timing to maintain momentum during weaker months.

## Visuals
![Monthly Revenue Trend](outputs/{relative_charts['monthly']})

![Top Revenue Products](outputs/{relative_charts['products']})

![Profit by Region](outputs/{relative_charts['region']})

![Revenue by Category](outputs/{relative_charts['category']})
"""
    # Write the markdown content to disk
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_dashboard(summary: dict[str, object], charts: dict[str, str]) -> None:
    """
    Generate a simple HTML dashboard that displays key metrics and the chart images.

    The HTML is written to DASHBOARD_PATH and references images in the outputs/ folder.
    """
    relative_charts = {key: Path(path).name for key, path in charts.items()}
    html = f"""<!DOCTYPE html>
<html lang=\"en\"> 
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Business Sales Performance Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f7fb; color: #233142; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .header {{ background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white; padding: 24px; border-radius: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }}
    .card {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
    img {{ width: 100%; border-radius: 8px; background: white; padding: 6px; }}
    .section {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <div class=\"container\">"""

    html += "\n    <div class=\"header\">\n      <h1>Business Sales Performance Dashboard</h1>\n      <p>Client-ready overview of revenue, profit, product performance, and regional opportunity.</p>\n    </div>\n\n    <div class=\"cards\">\n      <div class=\"card\"><h3>Total Revenue</h3><p>${summary['total_revenue']:.2f}</p></div>\n      <div class=\"card\"><h3>Total Profit</h3><p>${summary['total_profit']:.2f}</p></div>\n      <div class=\"card\"><h3>Top Product</h3><p>{summary['top_product']}</p></div>\n      <div class=\"card\"><h3>Top Region</h3><p>{summary['top_region']}</p></div>\n    </div>\n\n    <div class=\"section\">\n      <h2>Executive Summary</h2>\n      <p>The strongest growth potential comes from scaling the top-performing product and strengthening campaigns in the most profitable region. Revenue is trending upward, which supports a focused growth plan.</p>\n    </div>\n\n    <div class=\"grid\">\n      <div class=\"section\">\n        <h3>Monthly Revenue Trend</h3>\n        <img src=\"outputs/{relative_charts['monthly']}\" alt=\"Monthly revenue trend\" />\n      </div>\n      <div class=\"section\">\n        <h3>Top Revenue Products</h3>\n        <img src=\"outputs/{relative_charts['products']}\" alt=\"Top revenue products\" />\n      </div>\n      <div class=\"section\">\n        <h3>Profit by Region</h3>\n        <img src=\"outputs/{relative_charts['region']}\" alt=\"Profit by region\" />\n      </div>\n      <div class=\"section\">\n        <h3>Revenue by Category</h3>\n        <img src=\"outputs/{relative_charts['category']}\" alt=\"Revenue by category\" />\n      </div>\n    </div>\n  </div>\n</body>\n</html>\n"

    # Write the final HTML to disk
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    """
    Orchestrator function that runs the whole analysis pipeline in order:
    - generate data (or load)
    - clean
    - create charts
    - build summary
    - write report and dashboard

    Prints locations of generated artifacts at the end.
    """
    # Here the original code called generate_sales_data() unconditionally; preserve that behavior
    df = generate_sales_data()
    clean_df = clean_data(df)
    charts = create_charts(clean_df)
    summary = build_summary(clean_df)
    write_report(summary, charts)
    write_dashboard(summary, charts)
    print("Sales analysis completed successfully.")
    print(f"Report: {REPORT_PATH}")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"Top product: {summary['top_product']}")
    print(f"Top region: {summary['top_region']}")
    print(f"Top category: {summary['top_category']}")


if __name__ == "__main__":
    main()
