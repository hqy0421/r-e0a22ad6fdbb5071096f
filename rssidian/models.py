from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

Base = declarative_base()

# Columns added after the initial schema. Existing databases are upgraded in
# place; SQLite's ALTER TABLE ADD COLUMN is enough for these nullable columns.
_SCHEMA_MIGRATIONS = {
    "articles": {
        "embedding_vector": "TEXT",
    },
    "feeds": {
        # HTTP conditional-request validators for the cached feed document.
        "http_etag": "TEXT",
        "http_last_modified": "TEXT",
        # The time window actually covered by the most recent complete read.
        # These are updated together with the validators above, and only when
        # a read completed successfully (HTTP 200 with a parseable document, or
        # a 304 confirming the cached document is still current).
        "last_complete_read_at": "DATETIME",
        "last_window_start": "DATETIME",
        # Human-readable outcome of the most recent fetch attempt.
        "last_fetch_status": "VARCHAR(32)",
        # Set when a read returned entries the per-round cap left unhandled.
        # While set, the stored validators do not describe the current
        # document, so the next round must read it unconditionally.
        "pending_backlog": "BOOLEAN DEFAULT 0",
    },
}


def _migrate_schema(engine) -> None:
    """Add columns that are missing from pre-existing databases."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table, columns in _SCHEMA_MIGRATIONS.items():
            if table not in table_names:
                continue
            existing = {col["name"] for col in inspector.get_columns(table)}
            for column, sql_type in columns.items():
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
        conn.commit()


class Feed(Base):
    """Model for RSS feed subscriptions."""
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    website_url = Column(String(512), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, default=datetime.utcnow)
    muted = Column(Boolean, default=False)
    muted_reason = Column(Text, nullable=True)  # Reason for muting the feed
    peer_through = Column(Boolean, default=False)  # Whether to peer through aggregator to origin article
    error_count = Column(Integer, default=0)  # Count of consecutive errors
    no_entries_count = Column(Integer, default=0)  # Count of consecutive checks with no entries
    last_error = Column(Text, nullable=True)

    # Per-source fetch state. The validators and the read window are one unit:
    # they are only advanced together after a complete read of the document.
    http_etag = Column(Text, nullable=True)  # ETag for If-None-Match
    http_last_modified = Column(Text, nullable=True)  # Last-Modified for If-Modified-Since
    last_complete_read_at = Column(DateTime, nullable=True)  # When the window was last confirmed
    last_window_start = Column(DateTime, nullable=True)  # Start of the confirmed read window
    last_fetch_status = Column(String(32), nullable=True)  # ok | not_modified | empty | error | capped
    pending_backlog = Column(Boolean, default=False)  # entries deferred by the per-round cap

    # Statistics
    article_count = Column(Integer, default=0)  # Total count of ingested articles
    avg_quality_score = Column(Float, nullable=True)  # Average quality score of articles
    quality_tier_counts = Column(Text, nullable=True)  # JSON string of quality tier counts {"S": 5, "A": 10, ...}
    
    # Relationships
    articles = relationship("Article", back_populates="feed", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Feed(id={self.id}, title='{self.title}', muted={self.muted}, peer_through={self.peer_through})>"


class Article(Base):
    """Model for articles from RSS feeds."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    title = Column(String(512), nullable=False)
    url = Column(String(512), nullable=False)
    guid = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    author = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    quality_tier = Column(String(1), nullable=True)  # S, A, B, C, D
    quality_score = Column(Integer, nullable=True)  # 1-100
    labels = Column(String(255), nullable=True)
    embedding_generated = Column(Boolean, default=False)
    embedding_vector = Column(Text, nullable=True)  # JSON-serialized embedding vector
    word_count = Column(Integer, nullable=True)
    jina_enhanced = Column(Boolean, default=False)  # Flag to indicate if content was fetched from Jina.ai
    
    # Relationships
    feed = relationship("Feed", back_populates="articles")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:30]}...', quality_tier='{self.quality_tier or 'None'}')>"


def init_db(db_path: str) -> Session:
    """Initialize database and return session."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    _migrate_schema(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_db_session(db_path: str) -> Session:
    """Get a database session."""
    engine = create_engine(f"sqlite:///{db_path}")
    _migrate_schema(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()