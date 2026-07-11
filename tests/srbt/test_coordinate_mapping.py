import torch

from lib.models.layers.srbt_hypotheses import (
    crop_cxcywh_to_image_xywh,
    image_xywh_to_crop_cxcywh,
)


def test_full_frame_crop_to_image_matches_tracker_formula():
    crop_box = torch.tensor([0.25, 0.75, 0.20, 0.10])

    image_box = crop_cxcywh_to_image_xywh(
        crop_box,
        resize_factor=2.0,
        patch_size=256,
        crop_center=(200.0, 150.0),
    )

    assert torch.allclose(
        image_box,
        torch.tensor([155.2, 175.6, 25.6, 12.8]),
        atol=1e-5,
    )


def test_crop_image_coordinate_mapping_round_trip():
    image_boxes = torch.tensor([
        [100.0, 60.0, 80.0, 40.0],
        [175.0, 125.0, 50.0, 50.0],
        [-10.0, 20.0, 30.0, 20.0],
    ])

    crop_boxes = image_xywh_to_crop_cxcywh(
        image_boxes,
        resize_factor=1.6,
        patch_size=256,
        crop_center=(200.0, 150.0),
    )
    restored = crop_cxcywh_to_image_xywh(
        crop_boxes,
        resize_factor=1.6,
        patch_size=256,
        crop_center=(200.0, 150.0),
    )

    assert torch.allclose(restored, image_boxes, atol=1e-4)


def test_mapping_preserves_tensor_device_dtype_and_leading_shape():
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.1], [0.1, 0.9, 0.05, 0.08]]],
        dtype=torch.float64,
    )

    mapped = crop_cxcywh_to_image_xywh(
        boxes,
        resize_factor=2.5,
        patch_size=320,
        crop_center=(640.0, 360.0),
    )
    restored = image_xywh_to_crop_cxcywh(
        mapped,
        resize_factor=2.5,
        patch_size=320,
        crop_center=(640.0, 360.0),
    )

    assert mapped.shape == boxes.shape
    assert mapped.dtype == torch.float64
    assert mapped.device == boxes.device
    assert torch.allclose(restored, boxes, atol=1e-10)


def test_odd_crop_size_and_padded_image_box_round_trip():
    image_boxes = torch.tensor([
        [-25.0, -10.0, 30.0, 20.0],
        [615.0, 350.0, 50.0, 40.0],
    ])

    crop_boxes = image_xywh_to_crop_cxcywh(
        image_boxes,
        resize_factor=1.7,
        patch_size=255,
        crop_center=(320.0, 180.0),
    )
    restored = crop_cxcywh_to_image_xywh(
        crop_boxes,
        resize_factor=1.7,
        patch_size=255,
        crop_center=(320.0, 180.0),
    )

    assert torch.allclose(restored, image_boxes, atol=1e-4)
