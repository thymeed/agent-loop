# Test image for `type: project` tasks. Built by agent_loop/sandbox.py, never by hand.
# The build is the only step with network access: the optional requirements.txt in the
# build context (written from the task file's validated `requirements:`) is installed here,
# and every test run afterwards uses --network none.
ARG BASE=python:3.12-slim
FROM ${BASE}
COPY requirements.txt /tmp/requirements.txt
COPY verify_pytest.py /tmp/verify_pytest.py
# pytest is pinned to 9.2.0 because it decides every pass and fail the harness reports. Only pytest is
# pinned: right after that install, pip freeze records its dependencies (pluggy, iniconfig, packaging,
# pygments) at the versions pip resolved, and the requirements install is held to those (-c). A
# requirement that needs another version of any of them fails the build; it cannot replace them.
#
# Keeping project code from replacing pytest or running at interpreter start-up:
#  - work.pth puts /work and /work/src on sys.path AFTER the standard library and site-packages,
#    so a project file named pytest.py cannot shadow the real one. The path is resolved when
#    python starts, by which time /work is mounted.
#  - Python imports `sitecustomize` after reading .pth files, so with /work on the path a project's
#    sitecustomize.py would run at start-up. An empty sitecustomize.py in site-packages occupies
#    the name first. That line runs before the check and before any test, so it also replaces a
#    requirement's own sitecustomize.py, which would otherwise run in every test process.
#  - Tests run with `python -I`, which ignores PYTHONPATH and leaves the working directory off
#    sys.path.
#  - The build's own Python steps (the snapshot, the work.pth line and the check) run with
#    `python -I -S`. -S skips site, so no .pth file and no sitecustomize.py from a requirement runs
#    in them. -I alone does not stop that.
RUN pip install --no-cache-dir pytest==9.2.0 \
 && pip freeze > /tmp/constraints.txt \
 && python -I -S /tmp/verify_pytest.py snapshot \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && python -I -S -c "import pathlib, site; d = pathlib.Path(site.getsitepackages()[0]); (d / 'work.pth').write_text('/work\n/work/src\n'); (d / 'sitecustomize.py').write_text('# empty on purpose: see the Dockerfile\n')"
# Last step, in its own layer: fail the build if a requirement overwrote or added a file of pytest or
# its dependencies, which pip allows silently and the constraints cannot see. The check compares the
# disk with the snapshot taken right after the pytest install, not with RECORD, which a wheel can
# replace. This catches accidents only: a hostile package can still run code in the tests through a
# .pth file, or shadow pytest from the interpreter's stdlib directory. See verify_pytest.py.
RUN python -I -S /tmp/verify_pytest.py check
