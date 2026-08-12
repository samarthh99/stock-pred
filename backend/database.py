import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

# Try to load CA from file or environment
ca_cert = None
if os.path.exists("ca.pem"):
    with open("ca.pem", "r") as f:
        ca_cert = f.read()
elif os.getenv("CA_CERT"):
    ca_cert = os.getenv("CA_CERT")

ssl_args = {}
if ca_cert:
    # Write to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
        f.write(ca_cert)
        ca_path = f.name
    ssl_args = {"ssl": {"ca": ca_path}}

# connect_timeout ensures a bad/unreachable DB fails fast (seconds) instead of
# hanging indefinitely - important because engine creation and the first
# connection attempt can happen during app startup.
connect_args = {**ssl_args, "connect_timeout": 10}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
