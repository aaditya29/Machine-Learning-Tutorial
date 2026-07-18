# Here we are importing the necessary modules from SQLAlchemy. The create_engine function is used to create a connection to the database, while MetaData, Table, Column, Integer, and String are used to define the structure of our database tables. MetaData is a container object that holds information about the database schema, Table represents a table in the database, Column represents a column in a table, and Integer and String are data types for the columns.
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
engine = create_engine('sqlite:///mydatabase.db', echo=True)
meta = MetaData()  # meta is an instance of the MetaData class, which is used to hold information about the database schema. It acts as a container for Table objects and provides methods to create and manage tables in the database.

people = Table(
    # Here we are defining a new table named 'people' using the Table class. We pass in the name of the table, the MetaData instance, and a list of Column objects that define the structure of the table. Each Column object specifies the name of the column, its data type, and any additional constraints (e.g., primary key).
    "people", meta,
    # This line defines a column named 'id' of type Integer, which is also the primary key for the table. The primary key is a unique identifier for each row in the table.
    Column("id", Integer, primary_key=True),
    # This line defines a column named 'name' of type String, which cannot be null (i.e., it must have a value for each row).
    Column('name', String, nullable=False),
    Column('age', Integer, nullable=False)  #
)
# This line creates the 'people' table in the database using the engine we created earlier. The create_all() method checks if the table already exists and creates it if it doesn't.
meta.create_all(engine)
