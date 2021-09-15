FROM python:3.7-alpine
ADD loadtoMongo.py /
ADD requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5001
CMD ["python", "loadtoMongo.py"]