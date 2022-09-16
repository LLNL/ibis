import pkg_resources

try:
    d = pkg_resources.get_distribution("uq-methods")
    __version__ = d.version
    metadata = list(d._get_metadata(d.PKG_INFO))
    __sha__ = None
    for meta in metadata:
        if "Summary:" in meta:
            __sha__ = meta.split("(sha: ")[-1][:-1]
            break
    if __sha__ is not None:
        __version__ += "."+__sha__
except Exception:
    __version__ = "???"
    __sha__ = None


__all__ = ["likelihoods", "mcmc_diagnostics", "plots", "mcmc", "filter"]