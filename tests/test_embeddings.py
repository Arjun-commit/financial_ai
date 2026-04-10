from fin_flow.storage.embeddings import HashingEmbedder, cosine


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    v1 = e.embed("Scale marketing spend by 20% in Q3")
    v2 = e.embed("Scale marketing spend by 20% in Q3")
    assert v1 == v2


def test_hashing_embedder_similarity_ordering():
    e = HashingEmbedder(dim=128)
    query = e.embed("marketing spend in Q3")
    near = e.embed("marketing spend target for Q3")
    far = e.embed("office chairs and monitor stands")
    s_near = cosine(query, near)
    s_far = cosine(query, far)
    assert s_near > s_far
    assert 0.0 <= s_far <= s_near <= 1.0 + 1e-9


def test_empty_text_is_safe():
    e = HashingEmbedder(dim=32)
    v = e.embed("")
    assert all(x == 0.0 for x in v)
    # cosine with zero vector is 0, not NaN
    assert cosine(v, e.embed("hello")) == 0.0
