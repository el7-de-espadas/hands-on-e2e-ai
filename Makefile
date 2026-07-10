ifeq ($(OS),Windows_NT)
    PATHSEP := ;
else
    PATHSEP := :
endif

export PYTHONPATH := $(CURDIR)/apps/api$(PATHSEP)$(CURDIR)/apps/api/src$(PATHSEP)$(CURDIR)$(PATHSEP)$(PYTHONPATH)

run-evals-retriever:
	uv sync
	uv run --env-file .env python -m evals.eval_retriever

run-docker-compose:
	uv sync
	docker compose up --build 

clean-notebook-outputs:
	juoyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

# run-evals-retriever:
# 	uv sync
#  	PYTHONPATH=${PWD}/apps/api:${PWD}apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever
# 	$env:PYTHONPATH="$PWD;$PWD\apps\api;$PWD\apps\api\src;$env:PYTHONPATH"; uv run --env-file .env python -m evals.eval_retriever
