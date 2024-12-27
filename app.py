from fastapi import (
    FastAPI,
    Form,
    Request,
    Response,
    File,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline

app = FastAPI()
# docs uploded by user and output generated will get saved in static folder.
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# This FastAPI endpoint allows users to upload a PDF file via an HTTP POST request. The uploaded file
# is saved to a server directory, and the response includes a success message and the file path.
@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    ## Explanation:
    # async def chat(...): Defines an asynchronous function chat to handle the request. Asynchronous functions allow non-blocking I/O operations, improving performance for concurrent requests.
    # request: Request:Provides access to the HTTP request object, allowing inspection of headers, query parameters, etc.
    # pdf_file: bytes = File(): Specifies a file input parameter. The uploaded file is expected to be in binary format (bytes). FastAPI's File() ensures proper handling of file uploads.
    # filename: str = Form(...): Specifies a string input parameter expected from a form field. The Form(...) dependency extracts the filename value sent in the form.

    # base_folder: Sets the folder path where uploaded files will be saved.
    # os.path.isdir(base_folder): Checks if the directory static/docs/ exists.
    # os.mkdir(base_folder):Creates the directory static/docs/ if it does not already exist.

    # pdf_filename: Combines the base folder path with the provided filename to create the full path for saving the file.
    # aiofiles.open(pdf_filename, "wb"): Opens the file asynchronously in write-binary mode ("wb"). aiofiles is used for non-blocking file operations in asynchronous FastAPI applications.
    # await f.write(pdf_file): Writes the binary content of the uploaded file (pdf_file) to the specified path.

    # json.dumps(...): Creates a JSON string with a success message and the file path.
    # jsonable_encoder(...): Encodes the JSON string into a format compatible with FastAPI responses.

    # Response(response_data): Constructs an HTTP response with the encoded JSON data.

    base_folder = "static/docs/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, "wb") as f:
        await f.write(pdf_file)

    response_data = jsonable_encoder(
        json.dumps({"msg": "success", "pdf_filename": pdf_filename})
    )

    res = Response(response_data)
    return res


def get_csv(file_path):

    ## Explanation:
    # open(output_file, "w", newline="", encoding="utf-8"): Opens (or creates) the output file in write mode ("w"), ensuring UTF-8 encoding and proper handling of newlines. Context manager (with): Ensures the file is properly closed after writing.
    # csv.writer(csvfile): Initializes a CSV writer object to write data to the file.
    # csv_writer.writerow(["Question", "Answer"]): Writes the header row with column names "Question" and "Answer".

    # csv_writer.writerow([question, answer]): Writes the question and its corresponding answer as a row in the CSV file.

    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = "static/output/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("---------------------------------\n\n")

            # Save answer to csv file
            csv_writer.writerow([question, answer])
    return output_file


@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
