from dotenv import load_dotenv

from graph import GraphFactory
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = GraphFactory().create()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/ask")
async def ask(task_description: str, generation_mode: str = "dependencies"):
    return graph.invoke({"task_description": task_description, "generation_mode": generation_mode})


if __name__ == "__main__":
    uvicorn.run(app)
