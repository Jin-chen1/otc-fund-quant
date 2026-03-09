"""缓存管理器。"""

from pathlib import Path
from typing import Any

from diskcache import Cache
from loguru import logger

from ...config.settings import CACHE_DIR


class CacheManager:
    """基于磁盘的缓存管理器。"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_dir))
        logger.info(f"缓存管理器初始化完成，缓存目录: {cache_dir}")

    def get(self, key: str) -> Any | None:
        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"缓存命中: {key}")
            return value
        except Exception as e:
            logger.error(f"缓存读取失败 {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300):
        try:
            self.cache.set(key, value, expire=ttl)
            logger.debug(f"缓存设置成功: {key}, TTL: {ttl}秒")
        except Exception as e:
            logger.error(f"缓存写入失败 {key}: {e}")

    def delete(self, key: str):
        try:
            self.cache.delete(key)
            logger.debug(f"缓存删除成功: {key}")
        except Exception as e:
            logger.error(f"缓存删除失败 {key}: {e}")

    def clear(self):
        try:
            self.cache.clear()
            logger.info("缓存已清空")
        except Exception as e:
            logger.error(f"缓存清空失败: {e}")

    def close(self):
        self.cache.close()
