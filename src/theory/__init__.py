__version__ = 'v0.0.1'


from .discriminator import solve_sdp_bound, cobyla_objective


__all__ = ["solve_sdp_bound", "cobyla_objective"]
