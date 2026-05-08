# Code Citations

## License: desconhecido
https://github.com/switchablenorms/DeepFashion2/blob/cc5dd5721cede8d5f0041143e764c4bb0ef5f8e2/deepfashion2_api/PythonAPI/deepfashion2_retrieval_test.py

```
(results_name, 'r') as f:
    results = json.loads(f.read())
    for i in results:
        box = i['query_bbox']
        query_box = [box[0],box[1],box[2]-box[0],box[3]-box[1]]
        box = np.array(i['gallery_bbox'])
        gallery_box = [box[:,0], box[:,1], box[:,2] - box[:,0], box[:,3] - box[:,1]]
        gallery_box = np.transpose(gallery_box,(1,0)).tolist()
        
        results_image_id_all.append(i['query_image_id'])
        results_query_score_all.append(i['query_score'])
        results_query_cls_all.append(i['query_cls'])
        results_query_box_all.append(query_box)
        results_gallery_id_all.append(i['gallery_image_id'])
        results_gallery_box_all.append(gellery_box)
f.close(
```

