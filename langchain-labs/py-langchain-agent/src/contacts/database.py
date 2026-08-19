
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from settings import DatabaseSettings

settings = DatabaseSettings()

engine = create_engine(
    settings.url,
    pool_pre_ping=True,
)

SessionFactory = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
)

def get_db():
    session: Session = SessionFactory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()