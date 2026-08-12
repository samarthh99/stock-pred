import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# ---------- New imports for lifespan ----------
from contextlib import asynccontextmanager
import requests
from pathlib import Path

# ---------- Local imports ----------
from database import SessionLocal, engine, Base
from models import User, TradingSession
from auth import verify_password, get_password_hash, create_access_token, decode_access_token

# ---------- Create tables ----------
Base.metadata.create_all(bind=engine)

# ---------- Custom Attention Layer ----------
class AttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    def build(self, input_shape):
        self.W = self.add_weight(name='att_W', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.matmul(x, self.W) + self.b
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=-1)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context, alpha

# ---------- Feature Engineering ----------
def engineer_features_no_lookahead(df):
    df = df.copy()
    for d in [1, 5, 10, 20]:
        df[f'Ret_{d}d'] = df['Close'].pct_change(d)
        df[f'Ret_{d}d_sq'] = df[f'Ret_{d}d'] ** 2
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
        df[f'MA_ratio_{w}'] = df['Close'] / df[f'MA_{w}']
        df[f'MA_diff_{w}'] = df['Close'] - df[f'MA_{w}']
        df[f'MA_slope_{w}'] = df[f'MA_{w}'].diff()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_slope'] = df['RSI_14'].diff()
    df['RSI_overbought'] = (df['RSI_14'] > 70).astype(int)
    df['RSI_oversold'] = (df['RSI_14'] < 30).astype(int)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    middle = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_high'] = middle + 2*std
    df['BB_low'] = middle - 2*std
    df['BB_width'] = df['BB_high'] - df['BB_low']
    df['BB_pct'] = (df['Close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'] + 1e-8)
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df['ATR_14'] = tr.rolling(14).mean()
    df['ATR_ratio'] = df['ATR_14'] / df['Close']
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_ratio'] = df['Volume'] / df['Vol_SMA'].replace(0, 1)
    df['Vol_spike'] = (df['Vol_ratio'] > 2).astype(int)
    df['Volatility_20'] = df['Ret_1d'].rolling(20).std()
    df['Volatility_ratio'] = df['Volatility_20'] / df['Volatility_20'].rolling(100).mean()
    for w in [20, 50]:
        df[f'Price_pos_{w}'] = (df['Close'] - df['Low'].rolling(w).min()) / \
                               (df['High'].rolling(w).max() - df['Low'].rolling(w).min() + 1e-8)
    df['Dist_MA_50'] = (df['Close'] - df['MA_50']) / df['MA_50'] * 100
    df['Dist_MA_200'] = (df['Close'] - df['MA_200']) / df['MA_200'] * 100
    df['ADX'] = abs(df['MACD'] - df['MACD_signal']) / (df['ATR_14'] + 1e-8)
    df['Sector_ret'] = df['Ret_1d'].rolling(20).mean()
    df['Relative_strength'] = df['Ret_1d'] - df['Sector_ret']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# ---------- Global variables (set later) ----------
model = None
scalers = None
feature_cols = None
ppo_agent = None

# ---------- LIFESPAN: Startup & Shutdown ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scalers, feature_cols, ppo_agent
    
    print("🚀 Starting up: downloading and loading models...")
    
    MODEL_DIR = Path("paper_results")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Correct raw download URLs
    HF_REPO_BASE = "https://huggingface.co/SamKulkarni/stock-models/resolve/main"
    MODEL_FILES = {
        "cnn_bilstm_att_model.keras": f"{HF_REPO_BASE}/cnn_bilstm_att_model.keras",
        "scalers.pkl": f"{HF_REPO_BASE}/scalers.pkl",
        "feature_cols.pkl": f"{HF_REPO_BASE}/feature_cols.pkl",
        "ppo_agent.zip": f"{HF_REPO_BASE}/ppo_agent.zip",
    }
    
    def download_file(url, dest):
        if dest.exists():
            print(f"✅ {dest.name} already exists, skipping download.")
            return
        print(f"⬇️ Downloading {dest.name} from Hugging Face...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded {dest.name}")
        except Exception as e:
            print(f"❌ Failed to download {dest.name}: {e}")
            raise
    
    for filename, url in MODEL_FILES.items():
        download_file(url, MODEL_DIR / filename)
    
    # Load models
    print("Loading models...")
    model = keras.models.load_model(str(MODEL_DIR / "cnn_bilstm_att_model.keras"),
                                    custom_objects={'AttentionLayer': AttentionLayer})
    with open(MODEL_DIR / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(MODEL_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    ppo_agent = PPO.load(str(MODEL_DIR / "ppo_agent.zip"))
    
    print("✅ All models loaded successfully.")
    yield
    print("🛑 Shutting down...")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Stock Prediction & Trading API",
    version="1.0",
    lifespan=lifespan  # Critical: attaches the startup/shutdown events
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---------- DB Dependency ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Pydantic Schemas ----------
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictRequest(BaseModel):
    symbol: str

class PredictResponse(BaseModel):
    symbol: str
    probability: float
    signal: str
    explanation: str
    as_of_date: str

class TradeRequest(BaseModel):
    symbol: str
    current_price: float
    balance: float
    shares: int
    idx: int

class TradeResponse(BaseModel):
    action: int
    action_name: str
    probability: float
    explanation: str

# ---------- Helper: get current user ----------
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---------- Helper: CNN probability ----------
def get_cnn_prob(symbol: str):
    global model, scalers, feature_cols
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=200)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    df_eng = engineer_features_no_lookahead(df)
    if len(df_eng) < 60:
        raise HTTPException(status_code=400, detail="Not enough data after engineering")
    raw = df_eng[feature_cols].values[-60:]
    scaler = scalers.get(symbol)
    if scaler is None:
        raise HTTPException(status_code=400, detail=f"No scaler for {symbol}. Available: {list(scalers.keys())}")
    scaled = scaler.transform(raw)
    input_tensor = scaled.reshape(1, 60, -1)
    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    return prob, df_eng

# ---------- Auth Endpoints ----------
@app.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    hashed = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# ---------- Prediction Endpoint ----------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, current_user: User = Depends(get_current_user)):
    symbol = request.symbol.upper()
    prob, df_eng = get_cnn_prob(symbol)
    signal = "BUY" if prob > 0.6 else "SELL" if prob < 0.4 else "HOLD"
    if signal == "BUY":
        explanation = f"Model expects {symbol} to outperform its sector over the next 5 days with {prob:.1%} confidence. Consider buying."
    elif signal == "SELL":
        explanation = f"Model expects {symbol} to underperform its sector over the next 5 days with {1-prob:.1%} confidence. Consider selling."
    else:
        explanation = f"Model sees no clear edge (probability {prob:.1%}) – hold your position."
    return PredictResponse(
        symbol=symbol,
        probability=prob,
        signal=signal,
        explanation=explanation,
        as_of_date=df_eng.index[-1].strftime("%Y-%m-%d")
    )

# ---------- Trading Endpoint ----------
@app.post("/trade", response_model=TradeResponse)
def trade(request: TradeRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    global ppo_agent
    if ppo_agent is None:
        raise HTTPException(status_code=503, detail="PPO agent not loaded yet")
    symbol = request.symbol.upper()
    
    session = db.query(TradingSession).filter(
        TradingSession.user_id == current_user.id,
        TradingSession.symbol == symbol
    ).first()
    if not session:
        session = TradingSession(user_id=current_user.id, symbol=symbol, balance=10000, shares=0, idx=0)
        db.add(session)
        db.commit()
        db.refresh(session)
    
    session.balance = request.balance
    session.shares = request.shares
    session.idx = request.idx
    db.commit()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=300)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    closes = df['Close'].values
    if len(closes) < 100:
        raise HTTPException(status_code=400, detail="Not enough historical prices")

    window = 100
    start_idx = len(closes) - window
    idx = request.idx
    if idx >= window:
        raise HTTPException(status_code=400, detail=f"idx must be < {window}")
    norm_base = closes[start_idx]
    price = closes[start_idx + idx]
    if abs(price - request.current_price) > 0.05 * price:
        price = request.current_price

    ret_1d = (price - closes[start_idx + idx - 1]) / (closes[start_idx + idx - 1] + 1e-8) if idx > 0 else 0.0
    ret_5d = (price - closes[start_idx + idx - 5]) / (closes[start_idx + idx - 5] + 1e-8) if idx >= 5 else 0.0

    if idx >= 20:
        prices_window = closes[start_idx:start_idx + idx + 1]
        if len(prices_window) > 1:
            returns = np.diff(prices_window) / prices_window[:-1]
            vol = np.std(returns[-20:])
        else:
            vol = 0.0
    else:
        vol = 0.0

    prob, _ = get_cnn_prob(symbol)

    initial_balance = 10000.0
    pos = (request.shares * price) / initial_balance
    bal = request.balance / initial_balance
    prog = idx / window
    portfolio = request.balance + request.shares * price
    port_ret = (portfolio - initial_balance) / initial_balance

    obs = np.array([
        price / norm_base,
        ret_1d,
        ret_5d,
        vol,
        pos,
        bal,
        prog,
        port_ret,
        prob,
        0.0
    ], dtype=np.float32)

    action, _ = ppo_agent.predict(obs, deterministic=True)
    action = int(action)
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_name = action_names[action]

    if action == 0:
        explanation = "PPO agent recommends HOLD. No clear action – maintain current holdings."
    elif action == 1:
        explanation = f"PPO agent recommends BUY based on current portfolio and model signal. The model sees upside potential."
    else:
        explanation = f"PPO agent recommends SELL based on current portfolio and model signal. The model sees downside risk or profit-taking opportunity."

    return TradeResponse(
        action=action,
        action_name=action_name,
        probability=prob,
        explanation=explanation
    )

# ---------- Health check ----------
@app.get("/health")
def health():
    global scalers
    if scalers is None:
        return {"status": "loading", "message": "Models are still loading"}
    return {"status": "ok", "symbols": list(scalers.keys())}
