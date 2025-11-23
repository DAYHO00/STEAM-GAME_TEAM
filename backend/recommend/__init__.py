def recommend_by_model(*args, **kwargs):

    return model_based.recommend_by_model(*args, **kwargs)


def recommend_by_user(*args, **kwargs):
 
    from .user_based import recommend_by_user as _impl
    return _impl(*args, **kwargs)


def recommend_by_item(*args, **kwargs):
    from .item_based import recommend_by_item as _impl
    return _impl(*args, **kwargs)


def recommend_by_user_advanced(*args, **kwargs):
    from .user_based_advanced import recommend_by_user_advanced as _impl
    return _impl(*args, **kwargs)


def recommend_by_item_advanced(*args, **kwargs):
    from .item_based_advanced import recommend_by_item_advanced as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "model_based",              
    "recommend_by_model",
    "recommend_by_user",
    "recommend_by_item",
    "recommend_by_user_advanced",
    "recommend_by_item_advanced",
]
