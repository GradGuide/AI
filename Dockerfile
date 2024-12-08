FROM debian:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get dist-upgrade -y

RUN apt-get install -y python3 python3-pip build-essential

WORKDIR /app

COPY requirements.txt .

RUN pip install --break-system-packages --upgrade pip \
    && pip install --break-system-packages -r requirements.txt

COPY . .

RUN sh ./scripts/install.sh

EXPOSE 8000

CMD ["fastapi", "run", "scribe/api.py"]
