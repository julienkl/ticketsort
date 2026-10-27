FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit requests
COPY app ./app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
