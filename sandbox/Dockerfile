# Test image for `type: project` tasks. Built by agent_loop/sandbox.py, never by hand.
# The build is the only step with network access: the optional requirements.txt in the
# build context (written from the task file's validated `requirements:`) is installed here,
# and every test run afterwards uses --network none.
ARG BASE=python:3.12-slim
FROM ${BASE}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir pytest && pip install --no-cache-dir -r /tmp/requirements.txt
