#verificando se as imagens estão sendo pontuadas corretamente

#baixando as ferramentas e bibliotecas
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

#abrindo o arquivo e organizando as imagens
ROOT = Path(__file__).resolve().parents[2]
annFile = ROOT / 'evaluation' / 'example' / 'annotations.json'
bboxResFile = ROOT / 'evaluation' / 'example' / 'example_bbox_results.json'
keypointsResFile = ROOT / 'evaluation' / 'example' / 'example_keys_results.json'
segmResFile = ROOT / 'evaluation' / 'example' / 'example_segm_results.json'

for path in (annFile, bboxResFile, keypointsResFile, segmResFile):
    if not path.exists():
        raise FileNotFoundError(f"Required evaluation file not found: {path}")

cocoGt = COCO(str(annFile))
imgIds = sorted(cocoGt.getImgIds())

cocoDtBBox = cocoGt.loadRes(str(bboxResFile))
cocoDtKeypoints = cocoGt.loadRes(str(keypointsResFile))
# cocoDtSegm = cocoGt.loadRes(str(segmResFile))  # Commented out due to numpy compatibility issue

# evaluating clothes detection/ Avaliando a DETECÇÃO de roupas
cocoEval = COCOeval(cocoGt, cocoDtBBox, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# evaluating landmark and pose Estimation / pose
cocoEval = COCOeval(cocoGt, cocoDtKeypoints, 'keypoints')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# evaluating clothes segmentation
# cocoEval = COCOeval(cocoGt, cocoDtSegm, 'segm')
# cocoEval.params.imgIds = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()