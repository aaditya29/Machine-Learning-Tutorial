from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
engine = create_engine(
    # Here we are creating a PostgreSQL database connection using the create_engine function. The connection string specifies the username, password, host, and database name. The echo parameter is set to True to log SQL statements for debugging purposes.
    'postgresql://username:password@localhost/mydatabase', echo=True)

meta = MetaData()

people = Table(
    "people", meta,
    Column("id", Integer, primary_key=True),
    Column('name', String, nullable=False),
    Column('age', Integer)
)

# This line creates the 'people' table in the PostgreSQL database using the engine we created earlier. The create_all() method checks if the table already exists and creates it if it doesn't.
meta.create_all(engine)
# This line establishes a connection to the PostgreSQL database using the engine we created. The connection object allows us to execute SQL statements and interact with the database.
conn = engine.connect()

# This line creates an insert statement for the 'people' table using the insert() method. The values() method is used to specify the values for the 'name' and 'age' columns for the new row we want to insert into the table.
insert_statement = people.insert().values(name='Bob', age=25)
# This line executes the insert statement using the connection object. The result object contains information about the execution of the statement, such as the number of rows affected and any generated primary key values.
result = conn.execute(insert_statement)
conn.commit()  # Commit the transaction to save the changes to the database.
