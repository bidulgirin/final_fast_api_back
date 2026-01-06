from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool
from dotenv import load_dotenv

# Base metadata 가져오기
from app.db.base import Base

# 모델 등록: 개별 import 대신 한 방에 등록(누락/실수 방지)
# app/db/models/__init__.py 에서 모든 모델을 import해주면 가장 깔끔함
import app.db.models  # noqa: F401

# Alembic Config
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# .env 로드 및 DB URL 세팅
load_dotenv()
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

# alembic.ini의 sqlalchemy.url을 런타임에 주입
config.set_main_option("sqlalchemy.url", db_url)

# autogenerate 대상 metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 변경 감지 옵션(추천)
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 변경 감지 옵션(추천)
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
