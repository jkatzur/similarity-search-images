from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import uvicorn
import glob
from docarray import Document, DocumentArray
import torch
import torchvision

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

root_dir = 'static/images/'

state = {}
state['docs'] = []
state['model'] = None

def preproc(d: Document):
    return (d.load_uri_to_image_tensor()
            .set_image_tensor_shape((341, 256))
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1,0))

images = [file for file in glob.iglob(root_dir + '**/*.jpg', recursive=True)]

@app.get("/")
async def search_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/query")
async def query(request: Request, query = Form(...)):
    filtered_images = [image for image in images if query in image]
    return templates.TemplateResponse("results.html", {"request": request, "query": query, "images": filtered_images})

@app.get("/similarity")
async def search_form(request: Request):
    return templates.TemplateResponse("pick_image.html", {"request": request, "images": images})

@app.get("/similarity-query")
def similarity_search(request: Request, query_doc: str):
    query_doc = Document(uri=query_doc) # You can retrieve this from Weaviate
    query_doc = preproc(query_doc)      # Use the Weaviate Python Client in the future to avoid this
    query_doc.embed(state['model'])
    top6 = state['docs'].find(query_doc, limit=6)
    return templates.TemplateResponse("results.html", {"request": request, "query": query_doc.uri, "images": [doc.uri for doc in top6[0]]})

@app.get("/doc-count")
def get_doc_count():
    return len(state['docs'])

@app.on_event("startup")
def get_state():
    state['docs'] = DocumentArray(storage='weaviate', config={'host': 'localhost', 'port': 8080, 'name': 'Image4'})
    state['model'] = torchvision.models.resnet50(pretrained=True)       # We should switch this over to clip and use the Weaviate version


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)