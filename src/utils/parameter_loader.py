import json


def load_gridsearch_parameters():
    with open("src/conf/gridsearch.json", "r", encoding="utf-8") as read_file:
        data = json.load(read_file)
        TEST_FRACTION = data["TEST_FRACTION"]
        CV = data["CV"]
        OPT_ON = data["OPT_ON"]
        N_JOBS = data["N_JOBS"]
    return TEST_FRACTION, CV, OPT_ON, N_JOBS


def load_gridsearch_model_parameters():
    with open("src/conf/gridsearch_model_params.json", encoding="utf-8") as read_file:
        data = json.load(read_file)
    return data
