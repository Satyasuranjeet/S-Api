import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import motor.motor_asyncio
from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, UUID4
from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# Settings
# ===========================================================================
class Settings(BaseSettings):
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "secure_api"
    MONGODB_COLLECTION: str = "transactions"

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
# Schemas
# ===========================================================================
class TransactionCreate(BaseModel):
    id: UUID4
    type: str
    amount: float
    currency: str
    account_number: str
    sender: str
    merchant: str
    reference_number: str
    available_balance: float
    received_at: datetime
    raw_message: str


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
# Lifespan — connect / disconnect MongoDB
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mongo_client, _collection
    _mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    db = _mongo_client[settings.MONGODB_DB]
    _collection = db[settings.MONGODB_COLLECTION]

    await _collection.create_index("id", unique=True)
    logger.info("MongoDB connected")
    yield
    _mongo_client.close()


# ===========================================================================
# App
# ===========================================================================
app = FastAPI(title="Transaction API", version="1.0.0", lifespan=lifespan)


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
@app.post("/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    payload: TransactionCreate,
    col: motor.motor_asyncio.AsyncIOMotorCollection = Depends(get_collection),
) -> TransactionResponse:
    doc = {
        "id": str(payload.id),
        "type": payload.type,
        "amount": payload.amount,
        "currency": payload.currency,
        "account_number": payload.account_number,
        "sender": payload.sender,
        "merchant": payload.merchant,
        "reference_number": payload.reference_number,
        "available_balance": payload.available_balance,
        "received_at": payload.received_at,
        "raw_message": payload.raw_message,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        await col.insert_one(doc)
    except Exception as exc:
        if "duplicate key" in str(exc).lower() or "E11000" in str(exc):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Transaction ID already exists")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save transaction")

    return _serialize(doc)


# ===========================================================================
# GET /transactions
# ===========================================================================
@app.get("/transactions", response_model=PaginatedResponse)
async def get_transactions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    col: motor.motor_asyncio.AsyncIOMotorCollection = Depends(get_collection),
) -> PaginatedResponse:
    total = await col.count_documents({})
    cursor = col.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)
    return PaginatedResponse(total=total, skip=skip, limit=limit, data=[_serialize(d) for d in docs])
