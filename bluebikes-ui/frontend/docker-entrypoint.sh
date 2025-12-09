#!/bin/sh
# Replace PORT in nginx.conf if provided (Cloud Run provides PORT env var)
# Note: nginx.conf is already hardcoded to 8080 in my previous step, 
# but it's good practice to allow dynamic port.
# However, modifying the conf file at runtime is a bit hacky. 
# Cloud Run contract is to listen on $PORT. 
# Simplest approach: configure nginx to listen on 8080 and tell Cloud Run to use 8080 (default).
# So we don't strictly need a complex entrypoint if we stick to 8080.

exec "$@"
