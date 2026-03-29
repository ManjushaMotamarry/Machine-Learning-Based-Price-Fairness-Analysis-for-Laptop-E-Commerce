from fastapi import FastAPI

app = FastAPI()

items = []

@app.get("/")
def root():
    return {'Hello':"world"}

@app.get_item("/items{item_id}")
def get_items(item_id:int):
    item = items[item_id]
    return item

@app.post("/items")
def create_item(item:str):
    items.append(item)
    return items