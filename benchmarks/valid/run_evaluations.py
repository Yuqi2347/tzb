#!/usr/bin/env python
# encoding: utf-8
print("[DEBUG] Script start")
import argparse
import getopt
import hashlib
import io
import json
import os
import pylab
import sys

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class CocoResFormat:
  def __init__(self):
    self.res = []
    self.caption_dict = {}

  def read_multiple_files(self, filelist, hash_img_name):
    for filename in filelist:
      print( 'In file %s\n' % filename)
      self.read_file(filename, hash_img_name)

  def read_file(self, filename, hash_img_name):
    count = 0
    with open(filename,'r') as opfd:
      for line in opfd:
        count +=1
        id_sent = line.strip().split('\t')
        if len(id_sent)>2:
          id_sent = id_sent[-2:]
        if len(id_sent) == 1:
          id_sent.append('None')
        if len(id_sent) != 2:
          print(f"Error line (split len={len(id_sent)}): {line}")

        assert len(id_sent) == 2
        if isinstance(id_sent[1], bytes):
            sent = id_sent[1].decode('ascii', 'ignore')
        else:
            sent = id_sent[1]

        if hash_img_name:
          img_id = int(int(hashlib.sha256(id_sent[0].encode('utf-8')).hexdigest(), 16) % sys.maxsize)
        else:  
          img = id_sent[0].split('_')[-1].split('.')[0]
          img_id = int(img)
        imgid_sent = {}
        
        if img_id in self.caption_dict:
          assert self.caption_dict[img_id] == sent
        else:
          self.caption_dict[img_id] = sent
          imgid_sent['image_id'] = img_id
          imgid_sent['caption'] = sent
          self.res.append(imgid_sent)
        if count%1000 == 0:
          print( 'Processed %d ...' % count)

  def dump_json(self, outfile):
    res = self.res
    with io.open(outfile, 'w', encoding='utf-8') as fd:
      fd.write(json.dumps(res,
         ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': ')))


def main():
    HASH_IMG_NAME = True
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", type=str, required=True,
                        help='File containing model-generated/hypothesis sentences.')
    parser.add_argument("-r", "--references", type=str, required=True,
                        help='JSON File containing references/groundtruth sentences.')
    args = parser.parse_args()
    prediction_file = args.inputfile
    reference_file = args.references
    json_predictions_file = '{0}.json'.format(prediction_file)

    print("[INFO] Start reading prediction file...")
    crf = CocoResFormat()
    crf.read_file(prediction_file, HASH_IMG_NAME)
    print(f"[INFO] Finished reading prediction file, total captions: {len(crf.res)}")

    print("[INFO] Dumping JSON predictions...")
    crf.dump_json(json_predictions_file)
    print(f"[INFO] Dumped JSON to {json_predictions_file}")

    print("[INFO] Loading COCO reference annotations...")
    coco = COCO(reference_file)
    print("[INFO] Loaded COCO references.")

    print("[INFO] Loading COCO results...")
    cocoRes = coco.loadRes(json_predictions_file)
    print("[INFO] Loaded COCO results.")

    print("[INFO] Creating COCOEvalCap object...")
    cocoEval = COCOEvalCap(coco, cocoRes)
    print("[INFO] Created COCOEvalCap object.")

    print("[INFO] Starting evaluation...")
    cocoEval.evaluate()
    print("[INFO] Finished evaluation.")

    print("[INFO] Scores:")
    for metric, score in cocoEval.eval.items():
        print(f'{metric}: {score:.3f}')


  
if __name__ == "__main__":
    main()
