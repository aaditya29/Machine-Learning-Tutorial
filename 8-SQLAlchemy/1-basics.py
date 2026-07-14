from sqlalchemy import create_engine, text

# Here we are creating a SQLite database named 'mydatabase.db' and enabling echo to log SQL statements.
engine = create_engine('sqlite:///mydatabase.db', echo=True)

# This line establishes a connection to the database using the engine we created. The connection object allows us to execute SQL statements and interact with the database.
conn = engine.connect()
conn.execute(
    text("CREATE TABLE IF NOT EXISTS users (name TEXT, age INTEGER)")
    # This line executes a SQL statement to create a table named 'users' with two columns: 'name' (of type TEXT) and 'age' (of type INTEGER). The "IF NOT EXISTS" clause ensures that the table is only created if it doesn't already exist in the database.
)
conn.commit()  # Commit the changes to the database to make them permanent.
