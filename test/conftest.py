import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="load_threshold",  # default mode if not provided
        help="Run mode for the test: save_threshold or load_threshold",
    )
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="If set, enable plotting",
    )


@pytest.fixture(scope="session")
def run_mode(request):
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def plot_mode(request):
    return request.config.getoption("--plot")
