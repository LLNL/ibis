import pkg_resources

try:
    d = pkg_resources.get_distribution("ibis")
    if d.version != '1.0.0':
        __version__ = '1.0.0'
    metadata = list(d._get_metadata(d.PKG_INFO))
    __sha__ = None
    for meta in metadata:
        if "Summary:" in meta:
            __sha__ = meta.split("(sha: ")[-1][:-1]
            break
    if __sha__ is not None:
        __version__ += "."+__sha__
    
    print(d.version)
except Exception:
    __version__ = "???"
    __sha__ = None


__all__ = ["likelihoods", "mcmc_diagnostics", "plots", "mcmc", "filter", "sensitivity", "pce_model"]
