from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
engine = create_engine('sqlite:///mydatabase.db', echo=True)

meta = MetaData()

people = Table(
    "people", meta,
    Column("id", Integer, primary_key=True),
    Column('name', String, nullable=False),
    Column('age', Integer, nullable=False)
)
meta.create_all(engine)

conn = engine.connect()
# This line creates a select statement for the 'people' table using the select() method. The where() method is used to specify a condition that filters the rows returned by the query. In this case, we are filtering for rows where the 'age' column is greater than 30.
select_statement = people.select().where(people.c.age > 30)
# This line executes the select statement using the connection object. The result object contains the rows returned by the query, which we can iterate over to access the data.
result = conn.execute(select_statement)
for row in result.fetchall():  # This line iterates over the rows returned by the query using a for loop. The fetchall() method retrieves all the rows from the result object and returns them as a list of tuples.
    # This line prints each row to the console. Each row is represented as a tuple, where each element corresponds to a column in the 'people' table.
    print(row)
