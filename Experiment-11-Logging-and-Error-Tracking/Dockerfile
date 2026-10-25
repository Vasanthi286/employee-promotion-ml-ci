
FROM python:3.11-slim

WORKDIR /app

COPY Experiment-11-Logging-and-Error-Tracking/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY Experiment-11-Logging-and-Error-Tracking/app.py ./app.py

COPY Experiment-9-GitHub-Actions/employee_promotion_model.pkl ./employee_promotion_model.pkl

EXPOSE 5000

CMD ["python", "app.py"]
