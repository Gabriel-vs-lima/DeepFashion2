import json
import os
from pathlib import Path
import numpy as np
from pycocotools import mask as maskUtils

thresh = 0.5
ROOT = Path(__file__).resolve().parents[2]

results_name = ROOT / 'evaluation' / 'example' / 'example_retrieval_results.json'
query_name = ROOT / 'evaluation' / 'query_gt.json'
gallery_name = ROOT / 'evaluation' / 'gallery_gt.json'

for path in (results_name, query_name, gallery_name):
    if not path.exists():
        raise FileNotFoundError(f"Required evaluation file not found: {path}")

# load retrieval results
results_image_id_all = []
results_query_score_all = []
results_query_cls_all = []
results_query_box_all = []
results_gallery_id_all = []
results_gallery_box_all = []

with open(results_name, 'r') as f:
    results = json.load(f)
    for i in results:
        box = i['query_bbox']
        query_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        gallery_box_array = np.array(i['gallery_bbox'])
        gallery_box = np.stack(
            [gallery_box_array[:, 0], gallery_box_array[:, 1], gallery_box_array[:, 2] - gallery_box_array[:, 0], gallery_box_array[:, 3] - gallery_box_array[:, 1]],
            axis=1
        ).tolist()

        results_image_id_all.append(i['query_image_id'])
        results_query_score_all.append(i['query_score'])
        results_query_cls_all.append(i['query_cls'])
        results_query_box_all.append(query_box)
        results_gallery_id_all.append(i['gallery_image_id'])
        results_gallery_box_all.append(gallery_box)

print("Results loaded successfully")

results_image_id_all = np.array(results_image_id_all)
results_query_score_all = np.array(results_query_score_all)
results_query_cls_all = np.array(results_query_cls_all)
results_query_box_all = np.array(results_query_box_all)
results_gallery_id_all = np.array(results_gallery_id_all)
results_gallery_box_all = np.array(results_gallery_box_all)

# load query ground truth
query_image_id_all = []
query_box_all = []
query_cls_all = []
query_style_all = []
query_pair_all = []

with open(query_name, 'r') as f:
    query = json.load(f)
    for i in query:
        box = i['bbox']
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        query_image_id_all.append(i['query_image_id'])
        query_box_all.append(box)
        query_cls_all.append(i['cls'])
        query_style_all.append(i['style'])
        query_pair_all.append(i['pair_id'])

query_image_id_all = np.array(query_image_id_all)
query_box_all = np.array(query_box_all)
query_cls_all = np.array(query_cls_all)
query_style_all = np.array(query_style_all)
query_pair_all = np.array(query_pair_all)

query_num = len(np.where(query_style_all > 0)[0])
query_id_real = np.unique(query_image_id_all)

gallery_image_id_all = []
gallery_box_all = []
gallery_style_all = []
gallery_pair_all = []

with open(gallery_name, 'r') as f:
    gallery = json.load(f)
    for i in gallery:
        box = i['bbox']
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        gallery_image_id_all.append(i['gallery_image_id'])
        gallery_box_all.append(box)
        gallery_style_all.append(i['style'])
        gallery_pair_all.append(i['pair_id'])

gallery_image_id_all = np.array(gallery_image_id_all)
gallery_box_all = np.array(gallery_box_all)
gallery_style_all = np.array(gallery_style_all)
gallery_pair_all = np.array(gallery_pair_all)

correct_num_1 = 0
correct_num_5 = 0
correct_num_10 = 0
correct_num_15 = 0
correct_num_20 = 0
miss_num = 0

for query_image_id in query_id_real:
    results_id_ind = np.where(results_image_id_all == query_image_id)[0]
    if len(results_id_ind) == 0:
        continue

    query_id_ind = np.where(query_image_id_all == query_image_id)[0]
    pair_id = query_pair_all[query_id_ind]
    assert len(np.unique(pair_id)) == 1
    pair_id = pair_id[0]

    results_id_score = results_query_score_all[results_id_ind]
    results_id_box = results_query_box_all[results_id_ind]
    results_id_cls = results_query_cls_all[results_id_ind]
    results_id_gallery_id = results_gallery_id_all[results_id_ind]
    results_id_gallery_box = results_gallery_box_all[results_id_ind]

    query_id_box = query_box_all[query_id_ind]
    query_id_cls = query_cls_all[query_id_ind]
    query_id_style = query_style_all[query_id_ind]

    is_crowd = np.zeros(len(query_id_box))
    iou_id = maskUtils.iou(results_id_box, query_id_box, is_crowd)
    iou_ind = np.argmax(iou_id, axis=1)

    for local_idx in range(len(query_id_ind)):
        style = query_id_style[local_idx]
        cls = query_id_cls[local_idx]
        if style <= 0:
            continue

        results_style_ind1 = np.where(iou_ind == local_idx)[0]
        results_style_ind2 = np.where(results_id_cls == cls)[0]
        results_style_ind = np.intersect1d(results_style_ind1, results_style_ind2)

        if len(results_style_ind) == 0:
            miss_num += 1
            continue

        results_score_style = results_id_score[results_style_ind]
        score_max_ind = np.argmax(results_score_style)
        results_style_query_ind = results_style_ind[score_max_ind]
        results_style_gallery_id = results_id_gallery_id[results_style_query_ind]
        results_style_gallery_box = results_id_gallery_box[results_style_query_ind]

        gt_gallery_ind1 = np.where(gallery_pair_all == pair_id)[0]
        gt_gallery_ind2 = np.where(gallery_style_all == style)[0]
        gt_gallery_ind = np.intersect1d(gt_gallery_ind1, gt_gallery_ind2)
        gt_gallery_image_id = gallery_image_id_all[gt_gallery_ind]
        gt_gallery_box = gallery_box_all[gt_gallery_ind]

        assert len(gt_gallery_ind) > 0

        if len(gt_gallery_ind) == 1:
            gt_gallery_image_id = np.array([gt_gallery_image_id[0]])

        topK_values = [1, 5, 10, 15, 20]
        correct_counts = [0, 0, 0, 0, 0]

        max_predictions = len(results_style_gallery_id) if np.ndim(results_style_gallery_id) > 0 else 1
        for k_id, topK in enumerate(topK_values):
            topK = min(topK, max_predictions)
            found = False
            for t in range(topK):
                if results_style_gallery_id[t] not in gt_gallery_image_id:
                    continue
                which_ind = np.where(gt_gallery_image_id == results_style_gallery_id[t])[0]
                crowd = np.zeros(len(which_ind))
                iou_style = maskUtils.iou([results_style_gallery_box[t]], gt_gallery_box[which_ind], crowd)
                if np.any(iou_style >= thresh):
                    correct_counts[k_id] = 1
                    break

        correct_num_1 += correct_counts[0]
        correct_num_5 += correct_counts[1]
        correct_num_10 += correct_counts[2]
        correct_num_15 += correct_counts[3]
        correct_num_20 += correct_counts[4]

print('top-1')
print(float(correct_num_1) / query_num if query_num else 0.0)
print('top-5')
print(float(correct_num_5) / query_num if query_num else 0.0)
print('top-10')
print(float(correct_num_10) / query_num if query_num else 0.0)
print('top-15')
print(float(correct_num_15) / query_num if query_num else 0.0)
print('top-20')
print(float(correct_num_20) / query_num if query_num else 0.0)


