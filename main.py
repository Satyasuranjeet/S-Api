import logging
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional

import motor.motor_asyncio
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, UUID4
from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# ===========================================================================
# Logging — never log sensitive field values
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Settings  (loaded from .env)
# ===========================================================================
class Settings(BaseSettings):
    # REQUIRED — set a long random value in .env
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(48))"
    API_KEY: str

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "secure_api"
    MONGODB_COLLECTION: str = "transactions"

    # Hard cap on incoming JSON body (bytes)
    MAX_BODY_BYTES: int = 10_240  # 10 KB

    # slowapi rate-limit strings
    RATE_LIMIT_POST: str = "30/minute"
    RATE_LIMIT_GET: str = "60/minute"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


# ===========================================================================
# MongoDB async client  (motor)
# ===========================================================================
_mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_collection: Optional[motor.motor_asyncio.AsyncIOMotorCollection] = None


def get_collection() -> motor.motor_asyncio.AsyncIOMotorCollection:
    if _collection is None:  # pragma: no cover
        raise RuntimeError("Database not initialised")
    return _collection


# ===========================================================================
# Auth  — X-API-Key header, constant-time comparison
# ===========================================================================
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> None:
    """
    secrets.compare_digest prevents timing-based key enumeration.
    A uniform 401 response avoids leaking whether the key length was correct.
    """
    if not secrets.compare_digest(
        api_key.encode("utf-8"), settings.API_KEY.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ===========================================================================
# Pydantic schemas
# ===========================================================================
class TransactionType(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


class TransactionCreate(BaseModel):
    id: UUID4
    type: TransactionType
    # Positive, ≤ 10 million, max 2 decimal places
    amount: Decimal = Field(gt=Decimal("0"), le=Decimal("10000000.00"))
    # Exactly 3 uppercase ISO currency letters
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    account_number: str = Field(min_length=2, max_length=50)
    sender: str = Field(min_length=1, max_length=255)
    merchant: str = Field(min_length=1, max_length=255)
    # Alphanumeric + hyphens/underscores only — prevents NoSQL / injection
    reference_number: str = Field(
        min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9\-_]+$"
    )
    available_balance: Decimal = Field(ge=Decimal("0"), le=Decimal("10000000.00"))
    received_at: datetime
    raw_message: str = Field(min_length=1, max_length=1000)

    model_config = {"str_strip_whitespace": True}

    @field_validator("account_number")
    @classmethod
    def validate_account_number(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9*Xx]{2,50}$", v):
            raise ValueError(
                "account_number must be alphanumeric or masked (e.g. XX1234)"
            )
        return v.upper()

    @field_validator("amount", "available_balance")
    @classmethod
    def validate_decimal_places(cls, v: Decimal) -> Decimal:
        if v != v.quantize(Decimal("0.01")):
            raise ValueError("Must have at most 2 decimal places")
        return v

    @field_validator("sender", "merchant", "raw_message")
    @classmethod
    def strip_control_chars(cls, v: str) -> str:
        # Remove null bytes and non-printable ASCII control characters
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)


class TransactionResponse(BaseModel):
    id: str
    type: str
    amount: float
    currency: str
    account_number: str
    sender: str
    merchant: str
    reference_number: str
    available_balance: float
    received_at: str
    raw_message: str
    created_at: str


class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    data: List[TransactionResponse]


# ===========================================================================
# Rate limiter  (per client IP)
# ===========================================================================
limiter = Limiter(key_func=get_remote_address)


# ===========================================================================
# Lifespan — connect / disconnect MongoDB
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mongo_client, _collection
    _mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    db = _mongo_client[settings.MONGODB_DB]
    _collection = db[settings.MONGODB_COLLECTION]

    # Unique index on `id` ensures idempotency at the database level
    await _collection.create_index("id", unique=True)
    logger.info("MongoDB connected — db=%s col=%s", settings.MONGODB_DB, settings.MONGODB_COLLECTION)
    yield

    _mongo_client.close()
    logger.info("MongoDB connection closed")


# ===========================================================================
# FastAPI application
# Swagger UI / ReDoc disabled — enable only in development via env vars
# ===========================================================================
app = FastAPI(
    title="Secure Transaction API",
    version="1.0.0",
    docs_url=None,       # set to "/docs"         in dev
    redoc_url=None,      # set to "/redoc"         in dev
    openapi_url=None,    # set to "/openapi.json"  in dev
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — locked down; add trusted origins only
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],           # e.g. ["https://your-frontend.example.com"]
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Security middleware — body-size guard + hardened response headers
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1. Reject oversized bodies before FastAPI reads them
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": "Request body exceeds maximum allowed size"},
                )
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid Content-Length header"},
            )

    response = await call_next(request)

    # 2. Harden every response
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=63072000; includeSubDomains; preload"
    )
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Strip server fingerprinting headers
    if "server" in response.headers:
        del response.headers["server"]
    if "x-powered-by" in response.headers:
        del response.headers["x-powered-by"]

    return response


# ===========================================================================
# Helper — strip MongoDB's internal _id before returning
# ===========================================================================
def _serialize(doc: dict) -> TransactionResponse:
    received = doc["received_at"]
    created = doc["created_at"]

    if isinstance(received, datetime) and received.tzinfo is None:
        received = received.replace(tzinfo=timezone.utc)
    if isinstance(created, datetime) and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    return TransactionResponse(
        id=doc["id"],
        type=doc["type"],
        amount=float(doc["amount"]),
        currency=doc["currency"],
        account_number=doc["account_number"],
        sender=doc["sender"],
        merchant=doc["merchant"],
        reference_number=doc["reference_number"],
        available_balance=float(doc["available_balance"]),
        received_at=received.isoformat() if isinstance(received, datetime) else str(received),
        raw_message=doc["raw_message"],
        created_at=created.isoformat() if isinstance(created, datetime) else str(created),
    )


# ===========================================================================
# GET /
# ===========================================================================
@app.get("/", status_code=status.HTTP_200_OK, include_in_schema=False)
async def root():
    return {"message": "API is working"}


# ===========================================================================
# POST /transactions
# ===========================================================================
@app.post(
    "/transactions",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
    summary="Record a new transaction",
)
@limiter.limit(settings.RATE_LIMIT_POST)
async def create_transaction(
    request: Request,       # required by slowapi
    payload: TransactionCreate,
    col: motor.motor_asyncio.AsyncIOMotorCollection = Depends(get_collection),
) -> TransactionResponse:
    transaction_id = str(payload.id)

    doc = {
        "id": transaction_id,
        "type": payload.type.value,
        "amount": str(payload.amount),          # store as string to preserve precision
        "currency": payload.currency,
        "account_number": payload.account_number,
        "sender": payload.sender,
        "merchant": payload.merchant,
        "reference_number": payload.reference_number,
        "available_balance": str(payload.available_balance),
        "received_at": payload.received_at,
        "raw_message": payload.raw_message,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        await col.insert_one(doc)
    except Exception as exc:
        # Duplicate key from the unique index on `id`
        if "duplicate key" in str(exc).lower() or "E11000" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A transaction with this ID already exists",
            )
        logger.error("DB insert failed: %s", type(exc).__name__)   # no exc details — avoid leaking internals
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist transaction",
        )

    logger.info(
        "Transaction stored | id=%s type=%s currency=%s",
        transaction_id,
        payload.type.value,
        payload.currency,
    )
    return _serialize(doc)


# ===========================================================================
# GET /transactions
# ===========================================================================
@app.get(
    "/transactions",
    response_model=PaginatedResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
    summary="Retrieve all transactions (paginated)",
)
@limiter.limit(settings.RATE_LIMIT_GET)
async def get_transactions(
    request: Request,       # required by slowapi
    skip: int = Query(default=0, ge=0, le=10_000, description="Records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    col: motor.motor_asyncio.AsyncIOMotorCollection = Depends(get_collection),
) -> PaginatedResponse:
    total = await col.count_documents({})
    cursor = col.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)

    return PaginatedResponse(
        total=total,
        skip=skip,
        limit=limit,
        data=[_serialize(d) for d in docs],
    )
