# Built on the local ultralytics CUDA image with the current ultralytics release.
#   docker build -t sc-train ~/sc-lab/train-img      (FROM ultralytics/ultralytics + pip install -U ultralytics)
#   docker build -t sam-zs  ~/sc-lab/zs
# WORKDIR is /app on purpose: the base image's /ultralytics checkout would shadow the upgraded package.
# The controller (server.py) never imports torch; worker.py does, in a short-lived subprocess per request.
FROM sc-train:latest
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart
WORKDIR /app
COPY zs_core.py worker.py server.py /app/
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
