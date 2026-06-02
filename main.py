"""
ML Analytics Dashboard — FastAPI Backend
Run: uvicorn backend.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
import pickle, json, os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Analytics Dashboard API",
    description="Sales analytics + ML predictions powered by FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_DIR = Path("backend/models")
DATA_PATH = "data/sales_data.csv"
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

with open(MODEL_DIR / "churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)
with open(MODEL_DIR / "revenue_model.pkl", "rb") as f:
    revenue_model = pickle.load(f)
with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open(MODEL_DIR / "metrics.json") as f:
    model_metrics = json.load(f)


# ── Helper ───────────────────────────────────────────────────────────────────
def encode(row: dict) -> dict:
    for col in ["region", "product", "channel"]:
        le = label_encoders[col]
        val = row.get(col, le.classes_[0])
        row[col + "_enc"] = int(le.transform([val])[0])
    return row


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(BASE.parent / "frontend" / "index.html")


@app.get("/api/summary")
def get_summary():
    return {
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "total_profit": round(float(df["profit"].sum()), 2),
        "total_orders": int(len(df)),
        "avg_satisfaction": round(float(df["customer_satisfaction"].mean()), 2),
        "churn_rate": round(float(df["churn"].mean() * 100), 2),
        "avg_discount": round(float(df["discount_pct"].mean() * 100), 2),
    }


@app.get("/api/revenue-over-time")
def revenue_over_time():
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    grp = tmp.groupby("month").agg(revenue=("revenue", "sum"), orders=("revenue", "count")).reset_index()
    grp = grp.sort_values("month")
    return grp.to_dict(orient="records")


@app.get("/api/by-region")
def by_region():
    grp = df.groupby("region").agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        orders=("revenue", "count"),
    ).reset_index()
    return grp.to_dict(orient="records")


@app.get("/api/by-product")
def by_product():
    grp = df.groupby("product").agg(
        revenue=("revenue", "sum"),
        units=("units_sold", "sum"),
    ).reset_index().sort_values("revenue", ascending=False)
    return grp.to_dict(orient="records")


@app.get("/api/by-channel")
def by_channel():
    grp = df.groupby("channel").agg(
        revenue=("revenue", "sum"),
        orders=("revenue", "count"),
    ).reset_index()
    return grp.to_dict(orient="records")


@app.get("/api/model-metrics")
def get_model_metrics():
    return model_metrics


# ── Prediction schemas ────────────────────────────────────────────────────────

class ChurnInput(BaseModel):
    region: str = "North"
    product: str = "Product A"
    channel: str = "Online"
    units_sold: int = 50
    unit_price: float = 100.0
    customer_age: int = 35
    customer_satisfaction: int = 3
    discount_pct: float = 0.1
    marketing_spend: float = 1000.0


class RevenueInput(BaseModel):
    region: str = "North"
    product: str = "Product A"
    channel: str = "Online"
    units_sold: int = 50
    unit_price: float = 100.0
    discount_pct: float = 0.1
    marketing_spend: float = 1000.0


@app.post("/api/predict/churn")
def predict_churn(data: ChurnInput):
    row = encode(data.dict())
    features = [[
        row["region_enc"], row["product_enc"], row["channel_enc"],
        data.units_sold, data.unit_price, data.customer_age,
        data.customer_satisfaction, data.discount_pct, data.marketing_spend,
    ]]
    prob = churn_model.predict_proba(features)[0][1]
    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": int(prob >= 0.5),
        "risk_level": "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low",
    }


@app.post("/api/predict/revenue")
def predict_revenue(data: RevenueInput):
    row = encode(data.dict())
    features = [[
        row["region_enc"], row["product_enc"], row["channel_enc"],
        data.units_sold, data.unit_price, data.discount_pct, data.marketing_spend,
    ]]
    pred = revenue_model.predict(features)[0]
    return {"predicted_revenue": round(float(pred), 2)}
