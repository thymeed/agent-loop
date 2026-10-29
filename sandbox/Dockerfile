# Test image for `type: project` tasks. Built by agent_loop/sandbox.py, never by hand.
# The build is the only step with network access: the optional requirements.txt in the
# build context (written from the task file's validated `requirements:`) is installed here,
# and every test run afterwards uses --network none.
ARG BASE=python:3.12-slim
FROM ${BASE}
COPY requirements.txt /tmp/requirements.txt
# pytest is pinned because it decides every pass and fail the harness reports.
#
# Keeping project code from replacing pytest or running at interpreter start-up:
#  - work.pth puts /work and /work/src on sys.path AFTER the standard library and site-packages,
#    so a project file named pytest.py cannot shadow the real one. The path is resolved when
#    python starts, by which time /work is mounted.
#  - Python imports `sitecustomize` after reading .pth files, so with /work on the path a project's
#    sitecustomize.py would run at start-up. An empty sitecustomize.py in site-packages occupies
#    the name first.
#  - Tests run with `python -I`, which ignores PYTHONPATH and leaves the working directory off
#    sys.path.
RUN pip install --no-cache-dir pytest==9.2.0 \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -c "import pathlib, site; d = pathlib.Path(site.getsitepackages()[0]); (d / 'work.pth').write_text('/work\n/work/src\n'); (d / 'sitecustomize.py').write_text('# empty on purpose: see the Dockerfile\n')"
