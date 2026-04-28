FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY market_signal_bot.py .

CMD ["python", "-u", "market_signal_bot.py"]
