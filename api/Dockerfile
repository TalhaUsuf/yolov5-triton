FROM python:3.9

# make dir app 
RUN mkdir /app
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# volumes will be already mounted for below command (in docker-compose file)

# COPY . .

# run the fast - api server
CMD [ "uvicorn" , "api.app:app", "--host", "0.0.0.0", "--port", "8005"]
# CMD [ "uvicorn" , "app:app", "--host", "0.0.0.0", "--port", "8005"]