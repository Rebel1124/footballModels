from sqlalchemy import create_engine, sql, MetaData
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import os


SEASON_TABLENAME = "footystats_seasons"
DB_URL = os.environ["DB_URL"]

# 1. Create engine and get_db
engine = create_engine(DB_URL)
get_db = sessionmaker(engine)


# --- Needed only if you want to use query builder 'sql.select(...).where(...)'
metadata = MetaData()
metadata.reflect(engine, only=[SEASON_TABLENAME])
table = metadata.tables[SEASON_TABLENAME]
# ---


def get_matches_from_db(season_id: int) -> list[dict]:
    all_matches = []
    limit = 600
    offset = 0
    while True:
        # 2. Build sql query
        sttm = (
            sql.select(table.c.league_matches)
            .where(table.c.season_id == season_id)
            .limit(limit)
            .offset(offset)
        )

        # You can use raw SQL instead of query builder
        # sttm = text("""
        #     SELECT * from ...
        # """)

        try:
            # 3. Use this to execute the query and get a list of data
            with get_db() as db:
                matches = list(db.scalars(sttm))
        except Exception as e:
            print(f"Failed to retrieve matches (season_id={season_id}) due to: {e}")
            break  # Change to 'raise' if you want to exit the program without further processing of available matches

        if not matches:
            break

        all_matches.extend(matches)
        offset += limit

    print(f"Retrieved {len(all_matches)} matches from db (season_id={season_id})")
    return all_matches
