# Alembic은 Base.metadata를 알아야 함

from sqlalchemy.orm import declarative_base
Base = declarative_base()