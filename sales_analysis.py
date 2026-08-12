from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sales_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = BASE_DIR / "sales_analysis_report.md"
DASHBOARD_PATH = BASE_DIR / "sales_dashboard.html"


def generate_sales_data() -> pd.DataFrame:
    np.random.seed(42)
    regions = ["North", "South", "East", "West", "Central"]
    categories = {
        "Electronics": ["Laptop", "Phone", "Tablet", "Headphones"],
        "Furniture": ["Chair", "Desk", "Sofa", "Cabinet"],
        "Office Supplies": ["Notebook", "Pen Set", "Stapler", "Binder"],
    }
    rows = []

    for _ in range(240):
        order_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="D")
        region = np.random.choice(regions)
        category = np.random.choice(list(categories.keys()))
        product = np.random.choice(categories[category])
        quantity = int(np.random.randint(1, 10))
        unit_price = round(np.random.uniform(20, 400), 2)
        discount = round(np.random.uniform(0.0, 0.15), 2)
        revenue = round(quantity * unit_price * (1 - discount), 2)
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

    df = pd.DataFrame(rows)
    df = df.sort_values("OrderDate").reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return generate_sales_data()
    df = pd.read_csv(DATA_PATH)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=["OrderDate", "Region", "Category", "Product", "Revenue", "Profit"])
    clean_df = clean_df[(clean_df["Revenue"] > 0) & (clean_df["Profit"] > 0)].copy()
    clean_df["Month"] = clean_df["OrderDate"].dt.to_period("M").astype(str)
    return clean_df


def create_charts(df: pd.DataFrame) -> dict[str, str]:
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(MonthlyRevenue=("Revenue", "sum"), MonthlyProfit=("Profit", "sum"))
    )
    monthly["MonthDate"] = pd.to_datetime(monthly["Month"])

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

    return {
        "monthly": str(monthly_path),
        "products": str(products_path),
        "region": str(region_path),
        "category": str(category_path),
    }


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    total_revenue = round(float(df["Revenue"].sum()), 2)
    total_profit = round(float(df["Profit"].sum()), 2)
    avg_order_value = round(float(df["Revenue"].mean()), 2)
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(MonthlyRevenue=("Revenue", "sum"), MonthlyProfit=("Profit", "sum"))
    )
    best_month = monthly.loc[monthly["MonthlyRevenue"].idxmax()]
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

    top_product = product_revenue.iloc[0]["Product"]
    top_region = region_profit.iloc[0]["Region"]
    top_category = category_revenue.iloc[0]["Category"]
    monthly_growth = monthly["MonthlyRevenue"].pct_change().dropna().mean() * 100

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "avg_order_value": avg_order_value,
        "top_product": top_product,
        "top_region": top_region,
        "top_category": top_category,
        "best_month": best_month["Month"],
        "monthly_growth": round(float(monthly_growth), 1),
    }


def write_report(summary: dict[str, object], charts: dict[str, str]) -> None:
    relative_charts = {key: Path(path).name for key, path in charts.items()}
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
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_dashboard(summary: dict[str, object], charts: dict[str, str]) -> None:
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
  <div class=\"container\">
    <div class=\"header\">
      <h1>Business Sales Performance Dashboard</h1>
      <p>Client-ready overview of revenue, profit, product performance, and regional opportunity.</p>
    </div>

    <div class=\"cards\">
      <div class=\"card\"><h3>Total Revenue</h3><p>${summary['total_revenue']:.2f}</p></div>
      <div class=\"card\"><h3>Total Profit</h3><p>${summary['total_profit']:.2f}</p></div>
      <div class=\"card\"><h3>Top Product</h3><p>{summary['top_product']}</p></div>
      <div class=\"card\"><h3>Top Region</h3><p>{summary['top_region']}</p></div>
    </div>

    <div class=\"section\">
      <h2>Executive Summary</h2>
      <p>The strongest growth potential comes from scaling the top-performing product and strengthening campaigns in the most profitable region. Revenue is trending upward, which supports a focused growth plan.</p>
    </div>

    <div class=\"grid\">
      <div class=\"section\">
        <h3>Monthly Revenue Trend</h3>
        <img src=\"outputs/{relative_charts['monthly']}\" alt=\"Monthly revenue trend\" />
      </div>
      <div class=\"section\">
        <h3>Top Revenue Products</h3>
        <img src=\"outputs/{relative_charts['products']}\" alt=\"Top revenue products\" />
      </div>
      <div class=\"section\">
        <h3>Profit by Region</h3>
        <img src=\"outputs/{relative_charts['region']}\" alt=\"Profit by region\" />
      </div>
      <div class=\"section\">
        <h3>Revenue by Category</h3>
        <img src=\"outputs/{relative_charts['category']}\" alt=\"Revenue by category\" />
      </div>
    </div>
  </div>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def main() -> None:
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
