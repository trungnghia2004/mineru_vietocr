import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import math
import time
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import math
import time
from loguru import logger

class TextRecognizer:
    def __init__(self, args, **kwargs):
        # Load VietOCR configuration
        self.device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        vietocr_config = Cfg.load_config_from_name('vgg_transformer')
        vietocr_config['device'] = self.device
        vietocr_config['cnn']['pretrained'] = True
        vietocr_config['predictor']['beamsearch'] = False
        self.text_recognizer = Predictor(vietocr_config)
        
        # Batch processing parameters
        self.rec_batch_num = args.rec_batch_num if hasattr(args, 'rec_batch_num') else 6

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict"):
        """Process a list of images with VietOCR."""
        img_num = len(img_list)
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0

        with tqdm(total=img_num, desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                
                starttime = time.time()
                # Convert numpy arrays to PIL Images for VietOCR
                pil_images = []
                for ino in range(beg_img_no, end_img_no):
                    img = img_list[ino]
                    if img is None or img.size == 0:
                        # Create a dummy PIL image for invalid inputs
                        pil_images.append(Image.new('RGB', (32, 32), color='white'))
                        continue
                    
                    # Convert numpy array to PIL Image
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img
                    
                    pil_images.append(pil_img)
                
                # Predict with VietOCR (no preprocessing needed)
                batch_results = []
                for pil_img in pil_images:
                    try:
                        text, prob = self.text_recognizer.predict(pil_img, return_prob=True)
                        if prob is None:
                            prob = 0.9
                    except Exception as e:
                        logger.error(f"VietOCR prediction failed: {str(e)}")
                        text, prob = '', 0.0
                    batch_results.append([text, prob])
                
                # Assign results back to original indices
                for rno, result in enumerate(batch_results):
                    rec_res[beg_img_no + rno] = result
                
                elapse += time.time() - starttime
                pbar.update(end_img_no - beg_img_no)

        return rec_res, elapse