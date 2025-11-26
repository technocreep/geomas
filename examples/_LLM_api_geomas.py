import time
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi import BackgroundTasks
import asyncio
import typer
import logging
#from LLM_API.ollama_LLM import process_message
import requests
import httpx
from ollama_example import build_paths, \
    initialize_global_rag, default_collection_targets, create_chat_session, \
        ingest_local_documents, answer_with_combined_context
import socket
import os
from pathlib import Path
from fastapi.responses import JSONResponse

from geomas.core.rag_modules.data_adapter import format_text_context
from geomas.core.logging.logger import get_logger


logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
app = typer.Typer(help="GEOMAS")
logger = get_logger()



def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


app = FastAPI(debug=True)
ip = str(get_ip())
CALLBACK_URL = f"http://{ip}:{8020}/webSock/send_message"
UPLOAD_ROOT = Path(__file__).parent / "data"

task_queue = asyncio.Queue()
test_chat_id = "demo-chat"
test_file = "test_1.mmd"
paths = build_paths(
    documents_dir="./data1/global/uploads",
    global_rag_dir="./data/global/.vector-store",
    chat_dir=f"./data/{test_chat_id}",
    uploads_dir=f"./data/{test_chat_id}/uploads",
    local_rag_dir=f"./data/{test_chat_id}/.vector-store",
)
initialize_global_rag(paths=paths)

async def worker():
    async with httpx.AsyncClient() as client:
        while True:
            data = await task_queue.get()
            curent_type = data[0]
            if curent_type == "message":
                _, message, chat_id, history, params = data
                
                query_kwargs = {"top_k": 5}
                settings_overrides: dict[str, object] = {"temperature": 0.2}
                if "temperature" in params:
                    settings_overrides["temperature"] = params["temperature"]
                logger.info(f"Check folders...")
                include_global = False
                reset_local_rag = True


                path = os.path.join(UPLOAD_ROOT, f"{chat_id}")
                logger.info(path)

                if not os.path.exists(path):
                    os.makedirs(path)
                    include_global = True

                path_db = os.path.join(path, ".vector-store")
                if not os.path.exists(path_db):
                    os.makedirs(path_db)
                
                path = os.path.join(path, "uploads")
                logger.info(path)

                if not os.path.exists(path):
                    os.makedirs(path)

                paths = build_paths(
                    documents_dir="./data1/global/uploads",
                    global_rag_dir="./data/global/.vector-store",
                    chat_dir=f"./data/{chat_id}",
                    uploads_dir=f"./data/{chat_id}/uploads",
                    local_rag_dir=f"./data/{chat_id}/.vector-store",
                )
                # create empty small db
                try:
                    logger.info(f"Processing message: {message}")
                    include_global = True
                    collection_targets = default_collection_targets(chat_id, paths=paths, include_global=include_global)
                    query_kwargs["scopes"] = collection_targets
                    logger.info(f"Databases: {query_kwargs['scopes']}")
                    with create_chat_session(
                        paths=paths,
                        chat_id=chat_id,
                        settings=settings_overrides,
                        reset_local_rag=reset_local_rag,
                    ) as api:
                        logger.info("Step 3/4: Ingesting uploads...")
                        logger.info("Step 4/4: Querying combined context... [1]")
                        response, context_rows = answer_with_combined_context(
                            api,
                            message,
                            chat_id=chat_id,
                            query_kwargs=query_kwargs,
                        )
                        #show_results(response, context_rows)
                    #result = run_ollama_workflow(message, settings=params)
                    files = []

                    for entry in context_rows:
                        files.append(entry['document'])
                        score = entry.get("score")
                        if isinstance(score, (int, float)):
                            score_display = f"{float(score):.3f}"
                        else:
                            score_display = str(score)
                        scope = entry.get("database_scope")
                        scope_suffix = f", scope={scope}" if scope else ""
                        logger.info(f"- {entry.get('document')} (score={score_display}{scope_suffix})")
                    
                    res = response.get('answer') or 'No answer returned.'
                    # Send result to callback URL
                    response = await client.post(CALLBACK_URL, json={
                            "text": res,
                            "chat_id": chat_id,
                            "files": files,
                            "params": params
                        })
                    logger.info(f"Sent result, got status {response.status_code}")
                finally:
                    task_queue.task_done()
            else:
                _, filename, chat_id, file, include_global = data
                try:
                    # upload file to db
                    paths = build_paths(
                        documents_dir="./data1/global/uploads",
                        global_rag_dir="./data/global/.vector-store",
                        chat_dir=f"./data/{chat_id}",
                        uploads_dir=f"./data/{chat_id}/uploads",
                        local_rag_dir=f"./data/{chat_id}/.vector-store",
                    )

                    #paths["uploads_dir"] = Path(f"data/{chat_id}/uploads/")
                    settings_overrides: dict[str, object] = {"temperature": 0.2}

                    with create_chat_session(
                        paths=paths,
                        chat_id=chat_id,
                        settings=settings_overrides,
                        reset_local_rag=False,
                    ) as api:
                        logger.info("Step 3/4: Ingesting uploads...")
                        ingest_local_documents(
                            api,
                            paths=paths,
                        )
                        response = await client.post(CALLBACK_URL, json={
                                "text": api.pipeline.database_pipeline.description,
                                "chat_id": chat_id,
                                "files": "",
                                "params": {},
                            })
                        logger.info(api.pipeline.database_pipeline.description)
                        logger.info(f"Step 3/4 complete.")
                            # except Exception as e:
                            #     logger.info(f"Error processing task: {e}")
                finally:
                    task_queue.task_done()


def process_task(name: str):
    time.sleep(5)  # simulate long task
    logger.info(f"Task finished for {name}")

@app.post("/process_message")
async def run_task(data=Body()):
    await task_queue.put(["message", data["message"], data["chat_id"], data["history"], data["params"]])
    return {"status": "queued", "message": data["message"], "params": data["params"]}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

@app.post("/uploadFile")
async def receive_file(chat_id: str = Form(...), filename: str = Form(...), file: UploadFile = None):
    # Save the uploaded file
    include_global = False

    path = os.path.join(UPLOAD_ROOT, f"{chat_id}")
    if not os.path.exists(path):
        include_global = True
        os.makedirs(path)
    path_db = os.path.join(path, ".vector-store")
    if not os.path.exists(path_db):
        os.makedirs(path_db)


    path = os.path.join(path, "uploads")
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_location = os.path.join(path, f"{filename}")
    with open(file_location, "wb") as f:
        f.write(await file.read())
    await task_queue.put(["file", filename, chat_id, file, include_global])
    return JSONResponse({
        "message": "File received successfully",
        "chat_id": chat_id,
    })



if __name__ == '__main__':
    ip = str(get_ip())
    logger.info(f"Starting... Current IP: {ip}")
    uvicorn.run(app, host=ip, port=8021)
