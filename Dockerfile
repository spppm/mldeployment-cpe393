FROM python:3.9-slim

WORKDIR /app

COPY app/ /app/
COPY app/model.pkl /app/model.pkl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9000

CMD ["python", "app.py"]
