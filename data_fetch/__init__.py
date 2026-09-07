"""NSE data-fetching modules. Lazy exports to defer I/O side effects."""

__all__ = ["fetch_option_chain", "fetch_nse_option_chain", "CACHE_FILE"]

_LAZY = {
    "fetch_option_chain": "data_fetch.module11_option_chain",
    "_fetch_raw_jugaad": "data_fetch.module11_option_chain",
    "CACHE_FILE": "data_fetch.module11_option_chain",
    "fetch_nse_option_chain": "data_fetch.module11b_nse_scraper",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(name)
