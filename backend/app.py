"""FastAPI endpoints for explainable transaction investigations."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.service import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, RiskService
from src.syndicai_v4.transaction_history import DuplicateTransactionError

logger = logging.getLogger("syndicai.audit")
MINIMUM_API_KEY_LENGTH = 32


def _api_key_configured() -> bool:
    return len(os.environ.get("SYNDICAI_API_KEY", "")) >= MINIMUM_API_KEY_LENGTH


def _api_key_matches(supplied_key: str) -> bool:
    configured_key = os.environ.get("SYNDICAI_API_KEY", "")
    return (
        len(configured_key) >= MINIMUM_API_KEY_LENGTH
        and hmac.compare_digest(
            supplied_key.encode("utf-8"),
            configured_key.encode("utf-8"),
        )
    )

app = FastAPI(
    title="SyndicAI V4",
    description=(
        "Explainable fraud-risk investigation API. Scores prioritize human review; "
        "they do not establish that a transaction is fraudulent."
    ),
    version="4.0.0",
)


@app.middleware("http")
async def api_key_and_request_audit(request: Request, call_next) -> Response:
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    path = request.url.path

    if path not in {"/health", "/docs", "/openapi.json", "/redoc"}:
        supplied_key = request.headers.get("X-API-Key", "")
        if not _api_key_configured():
            response = JSONResponse(
                status_code=503,
                content={"detail": "API key authentication is not configured"},
            )
        elif not _api_key_matches(supplied_key):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Valid X-API-Key authentication is required"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)

    duration_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
    logger.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
            separators=(",", ":"),
        )
    )
    return response


def _audit_action(
    request: Request | None,
    *,
    action: str,
    success: bool,
    **context: str | int | bool,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "investigation_action",
                "request_id": getattr(getattr(request, "state", None), "request_id", None),
                "action": action,
                "success": success,
                **context,
            },
            separators=(",", ":"),
        )
    )


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    del error
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request. Check required fields, allowed values, and field limits.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    logger.error(
        json.dumps(
            {
                "event": "request_failed",
                "request_id": getattr(request.state, "request_id", None),
                "error_type": type(error).__name__,
            },
            separators=(",", ":"),
        )
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "The request could not be completed.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@lru_cache(maxsize=1)
def _service() -> RiskService:
    return RiskService()


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(
        ge=0,
        strict=True,
        description="Zero-based row in the validated processed reference dataset",
    )
    model: Literal["A", "B", "C"] = "C"


class TransactionScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(
        ge=1,
        strict=True,
        description="Must be greater than the immutable reference dataset maximum step",
    )
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    nameOrig: str = Field(min_length=1, max_length=128)
    nameDest: str = Field(min_length=1, max_length=128)
    event_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: Literal["A", "B"] = "B"
    operating_point: Literal[
        "max_f1",
        "budget_0_10",
        "budget_0_25",
        "budget_0_50",
        "budget_1_00",
        "budget_2_00",
    ] = "max_f1"

    @field_validator("nameOrig", "nameDest", "event_id")
    @classmethod
    def identifiers_must_not_be_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Identifiers must not be blank")
        if any(ord(character) < 32 for character in normalized):
            raise ValueError("Identifiers must not contain control characters")
        return normalized


class InvestigationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    missing_count = sum(not path.is_file() for path in expected)
    return {
        "status": "ready" if missing_count == 0 else "artifacts_missing",
        "missing_artifact_count": missing_count,
        "api_key_configured": _api_key_configured(),
    }


@app.get("/models")
def models() -> dict[str, object]:
    try:
        return _service().model_summary()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Required model artifacts are unavailable") from error


@app.get("/operating_points")
def operating_points() -> dict[str, object]:
    """Expose Model B policies selected on validation and their historical test outcomes."""
    try:
        return {"model": "B", "operating_points": _service().alert_operating_points()}
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Validation/test artifacts for alert operating points are unavailable",
        ) from error


@app.get("/transaction_limits")
def transaction_limits() -> dict[str, int]:
    """Return the immutable reference boundary for new transaction submissions."""
    try:
        return {"reference_max_step": _service().reference_max_step}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Required reference data is unavailable") from error


@app.get("/live/status")
def live_status() -> dict[str, object]:
    """Return durable event counts and the next valid chronological step."""
    try:
        return _service().live_status()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Required live-scoring artifacts are unavailable",
        ) from error


@app.get("/live/events")
def live_events(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, object]:
    """Return recent live scores and their evidence for authenticated review."""
    try:
        service = _service()
        status = service.live_status()
        return {
            "event_count": status["event_count"],
            "events": service.recent_live_events(limit),
        }
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Required live-scoring artifacts are unavailable",
        ) from error


@app.get("/alerts")
def alerts(
    model: Literal["A", "B", "C"] = "C",
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, object]]:
    try:
        return _service().alert_queue(model, limit)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Required investigation artifacts are unavailable") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/transactions/{row_index}")
def transaction(
    row_index: int,
    request: Request,
    model: Literal["A", "B", "C"] = "C",
) -> dict[str, object]:
    try:
        result = _service().inspect(row_index, model)
        _audit_action(
            request,
            action="inspect_transaction",
            success=True,
            row_index=row_index,
            model=model,
        )
        return result
    except FileNotFoundError as error:
        _audit_action(
            request,
            action="inspect_transaction",
            success=False,
            row_index=row_index,
            model=model,
            reason="artifacts_unavailable",
        )
        raise HTTPException(status_code=503, detail="Required investigation artifacts are unavailable") from error
    except IndexError as error:
        _audit_action(
            request,
            action="inspect_transaction",
            success=False,
            row_index=row_index,
            model=model,
            reason="row_not_found",
        )
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        _audit_action(
            request,
            action="inspect_transaction",
            success=False,
            row_index=row_index,
            model=model,
            reason="invalid_model",
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _audit_action(
            request,
            action="inspect_transaction",
            success=False,
            row_index=row_index,
            model=model,
            reason=type(error).__name__,
        )
        raise


@app.post("/score")
def score(payload: ScoreRequest, request: Request) -> dict[str, object]:
    """Score a transaction selected from the validated reference dataset."""
    try:
        result = transaction(payload.row_index, request, payload.model)
    except HTTPException:
        _audit_action(
            request,
            action="score_reference_transaction",
            success=False,
            row_index=payload.row_index,
            model=payload.model,
            reason="request_rejected",
        )
        raise
    except Exception as error:
        _audit_action(
            request,
            action="score_reference_transaction",
            success=False,
            row_index=payload.row_index,
            model=payload.model,
            reason=type(error).__name__,
        )
        raise
    _audit_action(
        request,
        action="score_reference_transaction",
        success=True,
        row_index=payload.row_index,
        model=payload.model,
    )
    return result


@app.post("/score_transaction")
def score_transaction(
    payload: TransactionScoreRequest,
    request: Request,
) -> dict[str, object]:
    """Score a new event strictly after the immutable reference history."""
    try:
        result = _service().score_transaction(
            {
                "step": payload.step,
                "type": payload.type,
                "amount": payload.amount,
                "nameOrig": payload.nameOrig,
                "nameDest": payload.nameDest,
            },
            payload.model,
            event_id=payload.event_id,
            operating_point=payload.operating_point,
        )
        _audit_action(
            request,
            action="score_transaction",
            success=True,
            model=payload.model,
            step=payload.step,
            operating_point=payload.operating_point,
        )
        return result
    except DuplicateTransactionError as error:
        _audit_action(
            request,
            action="score_transaction",
            success=False,
            model=payload.model,
            step=payload.step,
            reason="duplicate_event_id",
        )
        raise HTTPException(status_code=409, detail="event_id has already been used") from error
    except FileNotFoundError as error:
        _audit_action(
            request,
            action="score_transaction",
            success=False,
            model=payload.model,
            step=payload.step,
            reason="artifacts_unavailable",
        )
        raise HTTPException(status_code=503, detail="Required scoring artifacts are unavailable") from error
    except ValueError as error:
        _audit_action(
            request,
            action="score_transaction",
            success=False,
            model=payload.model,
            step=payload.step,
            reason="invalid_request",
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _audit_action(
            request,
            action="score_transaction",
            success=False,
            model=payload.model,
            step=payload.step,
            reason=type(error).__name__,
        )
        raise


@app.put("/investigations/{row_index}")
def update_investigation(
    row_index: int,
    update: InvestigationUpdate,
    request: Request,
) -> dict[str, object]:
    try:
        service = _service()
        if row_index < 0 or row_index >= service.transactions.total_rows:
            raise IndexError(f"Transaction row index must be from 0 to {service.transactions.total_rows - 1}")
        saved = service.investigations.update(row_index, update.status, update.note)
        _audit_action(
            request,
            action="update_investigation",
            success=True,
            row_index=row_index,
            status=update.status,
        )
        return saved
    except FileNotFoundError as error:
        _audit_action(
            request,
            action="update_investigation",
            success=False,
            row_index=row_index,
            reason="artifacts_unavailable",
        )
        raise HTTPException(status_code=503, detail="Required investigation artifacts are unavailable") from error
    except IndexError as error:
        _audit_action(
            request,
            action="update_investigation",
            success=False,
            row_index=row_index,
            reason="row_not_found",
        )
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        _audit_action(
            request,
            action="update_investigation",
            success=False,
            row_index=row_index,
            reason="invalid_update",
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _audit_action(
            request,
            action="update_investigation",
            success=False,
            row_index=row_index,
            reason=type(error).__name__,
        )
        raise
