"""
Sessions are the SQLAlchemy ORM's "handle" to the database. They manage the persistence operations for ORM-mapped objects, and they also provide a staging zone for all the objects loaded into the database session object during its lifespan. Persistence operations are operations that involve saving, updating, or deleting objects in the database. The session object is responsible for tracking changes to these objects and coordinating the necessary SQL statements to be executed against the database.
In this example, we will create a session and use it to add a new user to the database.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

engine = create_engine('sqlite:///mydatabase.db', echo=True)

# Create a new session object using the engine we created earlier.
session = Session(engine)
