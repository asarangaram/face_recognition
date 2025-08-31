#!/bin/env python

from gevent import monkey
monkey.patch_all() 


from app import create_app, socketio

# Flask app
application = create_app(debug=True)  # uWSGI will use this

# Local dev: run with SocketIO server
if __name__ == "__main__":
    socketio.run(application, host="0.0.0.0", port=5000)
