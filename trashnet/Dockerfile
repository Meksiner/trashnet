FROM python:3.10
RUN apt-get update && apt-get install -y python3-tk tk libx11-6 libgl1-mesa-glx
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
