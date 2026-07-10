FROM python:3.13-slim
WORKDIR /app
COPY populator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY populate_time_columns.py .
CMD ["python", "populate_time_columns.py"]
