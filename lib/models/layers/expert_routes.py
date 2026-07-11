def canonicalize_route(route, expert_names):
    """Return a route in expert-bank order."""
    if isinstance(route, str):
        route = (route,)
    if not isinstance(route, (list, tuple)):
        raise TypeError("route must be an expert name or a sequence of names")
    route = tuple(route)
    if not route:
        raise ValueError("route must contain at least one expert")
    if len(set(route)) != len(route):
        raise ValueError("route contains duplicate experts")

    order = {name: index for index, name in enumerate(expert_names)}
    unknown = [name for name in route if name not in order]
    if unknown:
        raise KeyError(f"unknown experts in route: {unknown}")
    return tuple(sorted(route, key=order.__getitem__))


def normalize_batch_routes(routes, expert_names, batch_size, default_expert):
    """Normalize a shared route or one route per batch sample."""
    if routes is None:
        shared = canonicalize_route((default_expert,), expert_names)
        return [shared] * batch_size
    if isinstance(routes, str):
        shared = canonicalize_route((routes,), expert_names)
        return [shared] * batch_size
    if isinstance(routes, tuple) and all(isinstance(name, str) for name in routes):
        shared = canonicalize_route(routes, expert_names)
        return [shared] * batch_size
    if not isinstance(routes, (list, tuple)):
        raise TypeError("routes must be a shared route or a per-sample sequence")
    if len(routes) != batch_size:
        raise ValueError("per-sample routes length must match batch size")
    return [canonicalize_route(route, expert_names) for route in routes]
