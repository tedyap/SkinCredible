FROM python:3.7
COPY api /deploy/api/
COPY model /deploy/model/
COPY requirements.txt /deploy/

WORKDIR /deploy/
RUN pip install -r requirements.txt


ENV PYTHONPATH /deploy
ENTRYPOINT [ "python" ]
CMD [ "api/app.py" ]