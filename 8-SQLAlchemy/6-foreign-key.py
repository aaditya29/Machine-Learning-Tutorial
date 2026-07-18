from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey, Float
engine = create_engine('sqlite:///mydatabase.db', echo=True)

meta = MetaData()

people = Table(
    "people", meta,
    Column("id", Integer, primary_key=True),
    Column('name', String, nullable=False),
    Column('age', Integer, nullable=False)
)

things = Table(
    "things", meta,
    Column("id", Integer, primary_key=True),
    Column('name', String, nullable=False),
    Column('price', Float),
    Column('owner_id', Integer, ForeignKey('people.id'))
)

meta.create_all(engine)

conn = engine.connect()
insert_people = people.insert().values([
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
])
insert_things = things.insert().values([
    {'name': 'Laptop', 'price': 1200.00, 'owner_id': 1},
    {'name': 'Phone', 'price': 800.00, 'owner_id': 2},
    {'name': 'Tablet', 'price': 500.00, 'owner_id': 1}
])
conn.execute(insert_people)
conn.commit()
conn.execute(insert_things)
conn.commit()
