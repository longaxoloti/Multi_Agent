from sqlalchemy import inspect, text
from storage.trusted_db import AgentDBRepository


def main() -> None:
    repo = AgentDBRepository()
    repo.initialize()
    inspector = inspect(repo.engine)

    if repo.engine.dialect.name == "postgresql":
        schemas = ["public", "system", "profile", "knowledge", "security"]
        tables: list[tuple[str, str]] = []
        for schema in schemas:
            for table_name in inspector.get_table_names(schema=schema):
                if table_name == "alembic_version":
                    continue
                tables.append((schema, table_name))

        with repo.engine.begin() as conn:
            for schema, table_name in tables:
                fq = f'"{schema}"."{table_name}"'
                conn.execute(text(f"TRUNCATE {fq} RESTART IDENTITY CASCADE"))

        with repo.engine.begin() as conn:
            nonzero = []
            for schema, table_name in tables:
                fq = f'"{schema}"."{table_name}"'
                count = int(conn.execute(text(f"SELECT COUNT(*) FROM {fq}")).scalar_one())
                if count != 0:
                    nonzero.append((schema, table_name, count))

        print(f"dialect={repo.engine.dialect.name}")
        print(f"tables_cleared={len(tables)}")
        print(f"nonzero_after={len(nonzero)}")
        if nonzero:
            print(nonzero)
    else:
        tables = [t for t in inspector.get_table_names() if t != "alembic_version"]

        with repo.engine.begin() as conn:
            for table_name in tables:
                conn.execute(text(f'DELETE FROM "{table_name}"'))

        with repo.engine.begin() as conn:
            nonzero = []
            for table_name in tables:
                count = int(conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one())
                if count != 0:
                    nonzero.append((table_name, count))

        print(f"dialect={repo.engine.dialect.name}")
        print(f"tables_cleared={len(tables)}")
        print(f"nonzero_after={len(nonzero)}")
        if nonzero:
            print(nonzero)


if __name__ == "__main__":
    main()
