from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import date, datetime
import yaml
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Load config.yaml
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
db_conf = config['database']

app = FastAPI()

class Article(BaseModel):
    id: int
    headline: str
    pub_date: Optional[date]
    thumbnail_url: Optional[str]
    article_url: str
    content_hash: str
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    positive_feedback: Optional[int]
    neutral_feedback: Optional[int]
    negative_feedback: Optional[int]
    categories: Optional[List[str]]
    created_at: Optional[datetime] 
    updated_at: Optional[datetime] 

def get_db_conn():
    return psycopg2.connect(
        host=db_conf['host'],
        port=db_conf['port'],
        dbname=db_conf['dbname'],
        user=db_conf['user'],
        password=db_conf['password']
    )

@app.get('/articles/', response_model=List[Article])
def get_articles(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    try:
        conn = get_db_conn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = f"SELECT * FROM {db_conf['table_name']}"
        conditions = []
        params = []
        if start_date:
            conditions.append("pub_date >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("pub_date <= %s")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post('/articles/{article_id}/feedback/')
def add_feedback(article_id: int, feedback_type: str = Query(..., regex="^(positive|neutral|negative)$")):
    if feedback_type not in ['positive', 'neutral', 'negative']:
        raise HTTPException(status_code=400, detail="Invalid feedback type.")
    column = f"{feedback_type}_feedback"
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        update_query = f"""
            UPDATE {db_conf['table_name']}
            SET {column} = {column} + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING id, {column};
        """
        cursor.execute(update_query, (article_id,))
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        if not result:
            raise HTTPException(status_code=404, detail="Article not found.")
        return {"article_id": result[0], "feedback_type": feedback_type, "new_count": result[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
