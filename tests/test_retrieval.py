import numpy as np

from lumen.retrieval import (
    aggregate_embeddings,
    compute_bidirectional_ranks,
    random_nonmatch_cosine,
)


def test_bidirectional_ranks_recover_row_matched_pairs_numpy():
    image = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    text = image.copy()

    out = compute_bidirectional_ranks(
        image,
        text,
        chunk_size=2,
        backend="numpy",
        group_ids=["slide_a", "slide_a", "slide_b", "slide_c"],
    )

    assert out["i2t"]["R@1"] == 1.0
    assert out["t2i"]["R@1"] == 1.0
    assert np.all(out["i2t_ranks"] == 1)
    assert np.all(out["t2i_ranks"] == 1)
    assert out["i2t_same_group"]["R@1"] == 1.0
    assert out["t2i_same_group"]["R@1"] == 1.0


def test_duplicate_images_are_deduplicated_for_text_to_image():
    # Two images, each repeated across two text rows (the ARCH-OPEN QA layout:
    # several captions per figure). Without dedup the duplicate image slots tie
    # the exact positive, so text->image same-group collapses onto the exact
    # metric and recall@k steps by the duplication factor.
    img_a = [1.0, 0.0, 0.0]
    img_b = [0.0, 1.0, 0.0]
    image = np.asarray([img_a, img_a, img_b, img_b])
    # Distinct captions; caption i is written for image row i.
    text = np.asarray([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.0],
        [0.2, 0.8, 0.0],
    ])

    out = compute_bidirectional_ranks(
        image,
        text,
        chunk_size=2,
        backend="numpy",
        group_ids=["fig_a", "fig_a", "fig_b", "fig_b"],
        image_ids=["a", "a", "b", "b"],
    )

    # The image candidate pool collapses to the two unique figures.
    assert out["n_image_candidates"] == 2
    assert out["candidate_pool_deduplicated"] is True
    # Every caption's own figure is its nearest unique image -> perfect recall,
    # and no duplicate image can occupy a spurious tied rank.
    assert np.all(out["t2i_ranks"] == 1)
    assert out["t2i"]["R@1"] == 1.0
    # Text candidates were already unique, so image->text is unchanged.
    assert out["n_text_candidates"] == 4


def test_random_nonmatch_cosine_never_samples_diagonal():
    image = np.eye(5, dtype=np.float32)
    text = image.copy()

    scores = random_nonmatch_cosine(image, text, n_samples=100, seed=7)

    assert scores.shape == (100,)
    assert np.all(scores == 0.0)


def test_aggregate_embeddings_pools_patch_rows_by_file_id():
    image = np.asarray([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    text = image.copy()
    rows = [
        {"pair_index": "0", "file_id": "slide_a", "wsi_id": "wsi_a", "caption": "a"},
        {"pair_index": "1", "file_id": "slide_a", "wsi_id": "wsi_a", "caption": "b"},
        {"pair_index": "2", "file_id": "slide_b", "wsi_id": "wsi_b", "caption": "c"},
    ]

    agg_image, agg_text, agg_rows = aggregate_embeddings(image, text, rows, field="file_id")

    assert agg_image.shape == (2, 3)
    assert agg_text.shape == (2, 3)
    assert [row["file_id"] for row in agg_rows] == ["slide_a", "slide_b"]
    assert [row["n_patches"] for row in agg_rows] == ["2", "1"]
    assert np.allclose(np.linalg.norm(agg_image, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(agg_text, axis=1), 1.0)
