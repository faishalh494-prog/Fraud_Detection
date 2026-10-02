"""FastAPI endpoints for explainable transaction investigations."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.service import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, RiskService

app = FastAPI(
    title="SyndicAI V4",
    description=(
        "Explainable fraud-risk investigation API. Scores prioritize human review; "
        "they do not establish that a transaction is fraudulent."
    ),
    version="4.0.0",
)


@lru_cache(maxsize=1)
def _service() -> RiskService:
    return RiskService()


class ScoreRequest(BaseModel):
    row_index: int = Field(ge=0, description="Zero-based row in the validated processed reference dataset")
    model: Literal["A", "B", "C"] = "C"


class TransactionScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, description="Positive PaySim simulation step")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    nameOrig: str = Field(min_length=1, max_length=128)
    nameDest: str = Field(min_length=1, max_length=128)
    model: Literal["A", "B"] = "B"

    @field_validator("nameOrig", "nameDest")
    @classmethod
    def account_id_must_not_be_whitespace(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Account identifiers must not be blank")
        return normalized


class InvestigationUpdate(BaseModel):
    status: Literal["Open", "Investigating", "Escalated", "Closed"]
    note: str = Field(default="", max_length=2000)


@app.get("/health")
def health() -> dict[str, object]:
    expected = [
        DEFAULT_DATA_DIR / "syndicai_v1_processed.parquet",
        DEFAULT_ARTIFACT_DIR / "model_bundle.joblib",
        DEFAULT_ARTIFACT_DIR / "metrics.json",
        DEFAULT_ARTIFACT_DIR / "network_features.parquet",
        *(DEFAULT_ARTIFACT_DIR / f"test_alerts_{name}.parquet" for name in MODEL_FEATURES),
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    return {
        "status": "ready" if not missing else "artifacts_missing",
        "missing_artifacts": missing,
    }


@app.get("/models")
def models() -> dict[str, object]:
    try:
        return _service().model_summary()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/alerts")
def alerts(
    model: Literal["A", "B", "C"] = "C",
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, object]]:
    try:
        return _service().alert_queue(model, limit)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/transactions/{row_index}")
def transaction(row_index: int, model: Literal["A", "B", "C"] = "C") -> dict[str, object]:
    try:
        return _service().inspect(row_index, model)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, object]:
    """Score a transaction selected from the validated reference dataset."""
    return transaction(request.row_index, request.model)


@app.post("/score_transaction")
def score_transaction(request: TransactionScoreRequest) -> dict[str, object]:
    """Score a new event using reference history strictly before its step."""
    try:
        return _service().score_transaction(
            {
                "step": request.step,
                "type": request.type,
                "amount": request.amount,
                "nameOrig": request.nameOrig,
                "nameDest": request.nameDest,
            },
            request.model,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.put("/investigations/{row_index}")
def update_investigation(
    row_index: int,
    update: InvestigationUpdate,
) -> dict[str, object]:
    try:
        service = _service()
        if row_index < 0 or row_index >= service.transactions.total_rows:
            raise IndexError(f"Transaction row index must be from 0 to {service.transactions.total_rows - 1}")
        return service.investigations.update(row_index, update.status, update.note)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
