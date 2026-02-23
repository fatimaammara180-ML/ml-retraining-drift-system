import pandas as pd
from sqlalchemy import create_engine

def load_df_from_postgres(table_name: str, db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return pd.read_sql_table(table_name, conn)

def save_df_to_postgres(df: pd.DataFrame, table_name: str, db_url: str, if_exists="append"):
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)