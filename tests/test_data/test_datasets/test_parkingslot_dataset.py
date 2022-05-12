import os.path as osp
import tempfile

import mmcv
import pytest

from mmdet.datasets import CocoParkingSlotDataset


def _create_ids_error_coco_parkingslot_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'ps_points': [50, 60, 20, 20],
        'angles': [90]
    }

    annotation_2 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'ps_points': [50, 60, 20, 20],
        'angles': [90]
    }

    categories = [{
        'id': 0,
        'name': '1',
        # 'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations': [annotation_1, annotation_2],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def test_coco_annotation_ids_unique():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_ids_error_coco_parkingslot_json(fake_json_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        CocoParkingSlotDataset(ann_file=fake_json_file, classes=('1', ), pipeline=[])

test_coco_annotation_ids_unique()