"""Database initialization script."""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy_utils import database_exists, create_database
from src.config import get_database_url

def init_vector_extension(db_url):
    """Initialize the vector extension in PostgreSQL."""
    conn = psycopg2.connect(db_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create vector extension if it doesn't exist
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    cur.close()
    conn.close()

def init_database(testing=False):
    """Initialize the database."""
    db_url = get_database_url(testing)
    
    # Create database if it doesn't exist
    if not database_exists(db_url):
        create_database(db_url)
    
    # Initialize vector extension
    init_vector_extension(db_url)
    
    print(f"{'Test' if testing else 'Main'} database initialized successfully!")

if __name__ == '__main__':
    # Initialize both main and test databases
    init_database(testing=False)
    init_database(testing=True) 