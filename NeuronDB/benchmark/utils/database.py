"""
Database connection and query execution utilities.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import RealDictCursor


class DatabaseManager:
    """
    Manages database connections and query execution for benchmarks.
    
    Supports both NeuronDB and pgvector extensions.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 5,
    ):
        """
        Initialize database manager.
        
        Args:
            connection_string: PostgreSQL connection string (DSN format)
            host: Database host (if not using connection_string)
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connection pool size
            max_connections: Maximum connection pool size
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # Build connection string from components
            parts = []
            if host:
                parts.append(f"host={host}")
            if port:
                parts.append(f"port={port}")
            if database:
                parts.append(f"dbname={database}")
            if user:
                parts.append(f"user={user}")
            if password:
                parts.append(f"password={password}")
            self.connection_string = " ".join(parts)
        
        # Create connection pool with retry logic
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                self.pool = pool.ThreadedConnectionPool(
                    min_connections,
                    max_connections,
                    self.connection_string
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise ConnectionError(f"Failed to create connection pool after {max_retries} attempts: {e}")
    
    def _get_connection(self, retries=3):
        """Get connection from pool with retry logic."""
        for attempt in range(retries):
            try:
                conn = self.pool.getconn()
                # Test connection is alive
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                except Exception:
                    # Connection is dead, close it and try again
                    try:
                        conn.close()
                    except:
                        pass
                    if attempt < retries - 1:
                        import time
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise
                return conn
            except Exception as e:
                if attempt < retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise ConnectionError(f"Failed to get connection after {retries} attempts: {e}")
    
    def _return_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def wait_for_recovery(self, max_wait: int = 60) -> bool:
        """
        Wait for database to exit recovery mode.
        
        Args:
            max_wait: Maximum seconds to wait
        
        Returns:
            True if database is ready, False if timeout
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_is_in_recovery()")
                        result = cur.fetchone()
                        if result and not result[0]:
                            # Database is ready
                            self._return_connection(conn)
                            return True
                finally:
                    self._return_connection(conn)
            except Exception:
                pass
            
            time.sleep(2)
        
        return False
    
    def ensure_extension(self, extension_name: str) -> bool:
        """
        Ensure PostgreSQL extension is installed.
        
        Args:
            extension_name: Name of extension (e.g., 'neurondb', 'vector')
        
        Returns:
            True if extension exists or was created, False otherwise
        """
        conn = self._get_connection()
        original_isolation = conn.isolation_level
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(
                        sql.Identifier(extension_name)
                    )
                )
            return True
        except Exception as e:
            print(f"Warning: Failed to create extension '{extension_name}': {e}")
            return False
        finally:
            # Reset isolation level before returning connection to pool
            try:
                conn.set_isolation_level(original_isolation)
            except:
                pass
            self._return_connection(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True,
        timing: bool = False,
        retries: int = 3
    ) -> Tuple[Optional[List[Dict[str, Any]]], float]:
        """
        Execute a query and return results with optional timing.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            timing: Whether to measure execution time
            retries: Number of retry attempts on failure
        
        Returns:
            Tuple of (results, execution_time_seconds)
        """
        last_error = None
        for attempt in range(retries):
            conn = None
            try:
                conn = self._get_connection()
                start_time = time.perf_counter()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    if fetch:
                        results = cur.fetchall()
                        # Convert RealDictRow to dict
                        results = [dict(row) for row in results]
                    else:
                        results = None
                    conn.commit()
                end_time = time.perf_counter()
                execution_time = end_time - start_time if timing else 0.0
                self._return_connection(conn)
                return results, execution_time
            except Exception as e:
                last_error = e
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                    try:
                        self._return_connection(conn)
                    except:
                        # Connection is bad, close it
                        try:
                            conn.close()
                        except:
                            pass
                
                # Check if database is in recovery mode
                error_str = str(e).lower()
                if 'recovery mode' in error_str:
                    if attempt < retries - 1:
                        import time
                        wait_time = 2.0 * (attempt + 1)
                        print(f"    Database in recovery mode, waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Database is in recovery mode. Please wait for recovery to complete: {e}")
                
                # For other errors, retry with backoff
                if attempt < retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Query execution failed after {retries} attempts: {e}")
        
        raise RuntimeError(f"Query execution failed: {last_error}")
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple],
        timing: bool = False
    ) -> float:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            timing: Whether to measure execution time
        
        Returns:
            Total execution time in seconds
        """
        conn = self._get_connection()
        try:
            start_time = time.perf_counter()
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
            conn.commit()
            end_time = time.perf_counter()
            return (end_time - start_time) if timing else 0.0
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Batch execution failed: {e}")
        finally:
            self._return_connection(conn)
    
    def drop_table_if_exists(self, table_name: str, schema: Optional[str] = None) -> None:
        """
        Drop a table if it exists.
        
        Args:
            table_name: Name of table to drop
            schema: Optional schema name
        """
        conn = self._get_connection()
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                if schema:
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                            sql.Identifier(schema),
                            sql.Identifier(table_name)
                        )
                    )
                else:
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table_name)
                        )
                    )
        finally:
            self._return_connection(conn)
    
    def drop_index_if_exists(self, index_name: str, schema: Optional[str] = None) -> None:
        """
        Drop an index if it exists.
        
        Args:
            index_name: Name of index to drop
            schema: Optional schema name
        """
        conn = self._get_connection()
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                if schema:
                    cur.execute(
                        sql.SQL("DROP INDEX IF EXISTS {}.{} CASCADE").format(
                            sql.Identifier(schema),
                            sql.Identifier(index_name)
                        )
                    )
                else:
                    cur.execute(
                        sql.SQL("DROP INDEX IF EXISTS {} CASCADE").format(
                            sql.Identifier(index_name)
                        )
                    )
        finally:
            self._return_connection(conn)
    
    def get_table_size(self, table_name: str, schema: Optional[str] = None) -> int:
        """
        Get table size in bytes.
        
        Args:
            table_name: Name of table
            schema: Optional schema name
        
        Returns:
            Table size in bytes
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if schema:
                    # Use format() with %I for identifier to properly handle regclass
                    cur.execute(
                        "SELECT pg_total_relation_size(%s.%s::regclass) as size",
                        (schema, table_name)
                    )
                else:
                    # Use format() with %I for identifier to properly handle regclass
                    cur.execute(
                        "SELECT pg_total_relation_size(%s::regclass) as size",
                        (table_name,)
                    )
                result = cur.fetchone()
                return result[0] if result else 0
        finally:
            self._return_connection(conn)
    
    def get_index_size(self, index_name: str, schema: Optional[str] = None) -> int:
        """
        Get index size in bytes.
        
        Args:
            index_name: Name of index
            schema: Optional schema name
        
        Returns:
            Index size in bytes
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if schema:
                    # Use format() with %I for identifier to properly handle regclass
                    cur.execute(
                        "SELECT pg_relation_size(%s.%s::regclass) as size",
                        (schema, index_name)
                    )
                else:
                    # Use format() with %I for identifier to properly handle regclass
                    cur.execute(
                        "SELECT pg_relation_size(%s::regclass) as size",
                        (index_name,)
                    )
                result = cur.fetchone()
                return result[0] if result else 0
        finally:
            self._return_connection(conn)
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if hasattr(self, 'pool'):
            self.pool.closeall()

