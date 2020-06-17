FROM python:3.7
COPY ./api /deploy/
COPY ./model /deploy/
COPY ./requirements.txt /deploy/

RUN pip install -r requirements.txt
WORKDIR /deploy/api

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]