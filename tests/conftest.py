import os
from pathlib import Path
import sys
import tempfile

import pytest

PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from otc_fund_quant.nav_estimator.data.cache.cache_manager import CacheManager
from otc_fund_quant.nav_estimator.data.fetcher.base_fetcher import BaseFetcher


@pytest.fixture(autouse=True)
def _isolated_nav_estimator_cache():
    original_cache = BaseFetcher.cache
    with tempfile.TemporaryDirectory(prefix="tmp_nav_estimator_cache_") as temp_dir:
        temp_cache = CacheManager(Path(temp_dir))
        BaseFetcher.cache = temp_cache
        try:
            yield
        finally:
            BaseFetcher.cache = original_cache
            temp_cache.close()
