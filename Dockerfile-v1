FROM python:3.14.2-slim-trixie

WORKDIR /app

COPY . .

RUN python3 -m pip install -r requirements.txt

# Default Streamlit port
EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
