import os

# Gunicorn configuration
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
errorlog = "-"
accesslog = "-"
loglevel = "info" 