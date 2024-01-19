from hashlib import blake2b

def get_hash(input : str, digest_size=4):
    return blake2b(
        bytes(input, 'utf-8'),
        digest_size=digest_size
    ).hexdigest()