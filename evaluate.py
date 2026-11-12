def split_ratings(ratings):
    cutoff = int(ratings["timestamp"].quantile(0.8))
    return cutoff, ratings[ratings["timestamp"] < cutoff], ratings[ratings["timestamp"] >= cutoff]
