#!/bin/env python

from gevent import monkey
monkey.patch_all() 


from app import create_app, socketio
import os

application = create_app(debug=True)

# Redis for inter-worker communication
redis_url = os.getenv("REDIS_URL", None)

if redis_url:
    socketio.init_app(application, cors_allowed_origins="*", message_queue=redis_url, async_mode="gevent")
else:
    socketio.init_app(application)

if __name__ == "__main__":
    socketio.run(application, host="0.0.0.0", port=5000)
