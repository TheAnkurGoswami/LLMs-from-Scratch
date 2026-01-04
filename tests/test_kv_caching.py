import pytest
import torch

from inference.kv_caching import KeyValueCaching


def test_constructor_with_list_sets_attributes_none():
    cache = KeyValueCaching(["k", "v"])
    assert hasattr(cache, "k")
    assert hasattr(cache, "v")
    assert getattr(cache, "k") is None
    assert getattr(cache, "v") is None


def test_prefill_and_decode_single_tensor():
    cache = KeyValueCaching(["k"])  # predeclare cache keys

    t1 = torch.randn(2, 1, 3)
    (ret1,) = cache.update(k=t1)

    # After prefill, attribute should equal the provided tensor
    assert torch.equal(ret1, t1)
    assert torch.equal(cache.k, t1)

    # Decoding stage: concatenate along dim=1
    t2 = torch.randn(2, 2, 3)
    (ret2,) = cache.update(k=t2)

    expected = torch.cat([t1, t2], dim=1)
    assert torch.equal(ret2, expected)
    assert torch.equal(cache.k, expected)

    # Ensure original tensors aren't altered
    assert torch.equal(t1, t1)
    assert torch.equal(t2, t2)


def test_multiple_tensors_and_return_order():
    cache = KeyValueCaching(["k", "v"])  # predeclare cache keys

    k1 = torch.ones(1, 1, 2)
    v1 = torch.zeros(1, 1, 2)

    # Prefill two tensors in a single call; maintain insertion order
    k_ret, v_ret = cache.update(k=k1, v=v1)
    assert torch.equal(k_ret, k1)
    assert torch.equal(v_ret, v1)

    # Append new tensors and verify concatenation per key
    k2 = torch.full((1, 2, 2), 3.0)
    v2 = torch.full((1, 2, 2), 4.0)

    k_ret2, v_ret2 = cache.update(k=k2, v=v2)

    assert torch.equal(k_ret2, torch.cat([k1, k2], dim=1))
    assert torch.equal(v_ret2, torch.cat([v1, v2], dim=1))


def test_concat_shape_mismatch_raises():
    cache = KeyValueCaching(["k"])  # predeclare cache keys

    t1 = torch.randn(2, 1, 3)
    (ret1,) = cache.update(k=t1)

    # Provide a tensor with incompatible batch size (dim=0); concatenation should fail
    t_bad = torch.randn(3, 2, 3)
    with pytest.raises(RuntimeError):
        cache.update(k=t_bad)


def test_empty_list_allows_dynamic_names():
    cache = KeyValueCaching([])  # no predeclared names

    t = torch.randn(1, 1, 4)
    (ret,) = cache.update(k=t)  # dynamic add
    assert torch.equal(ret, t)
    assert torch.equal(cache.k, t)
