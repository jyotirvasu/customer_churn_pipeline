FROM apache/airflow:2.9.0

USER root
RUN pip install scikit-learn pandas numpy

USER airflow

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /requirements.txt.
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code
COPY project_structure.py ./

# Default command (not used by Airflow since it specifies commands per task)
CMD ["python", "project_structure.py"]

