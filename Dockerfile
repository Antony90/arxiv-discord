FROM python:latest
WORKDIR /app/

COPY *.py requirements.txt ./
RUN python3 -m pip install -r requirements.txt
RUN mkdir logs

COPY ai/*.py ./ai/
COPY bot/*.py ./bot/

ENTRYPOINT [ "python3", "main.py" ]