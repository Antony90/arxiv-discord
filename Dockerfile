FROM python:latest
WORKDIR /app/

COPY *.py requirements.txt ./
COPY ai/*.py ./ai/
COPY bot/*.py ./bot/

RUN python3 -m pip install -r requirements.txt
RUN mkdir logs

ENTRYPOINT [ "python3", "main.py" ]