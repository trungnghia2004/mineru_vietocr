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
        vietocr_config['predictor']['beamsearch'] = False
        self.text_recognizer = Predictor(vietocr_config)
        
        # Batch processing parameters
        self.rec_batch_num = args.rec_batch_num if hasattr(args, 'rec_batch_num') else 6
        self.rec_image_shape = [3, 32, 512]  # Default VietOCR input shape: [channels, height, width]
        

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict"):
        """Process a list of images with VietOCR."""
        img_num = len(img_list)
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0

        # Sort images by aspect ratio to optimize batch processing
        width_list = [img.shape[1] / float(img.shape[0]) if img.shape[0] > 0 else 0 for img in img_list]
        indices = np.argsort(np.array(width_list))

        with tqdm(total=img_num, desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                pil_image_batch = []
                batch_indices = indices[beg_img_no:end_img_no]
                
                # Preprocess images for the batch
                for ino in batch_indices:
                    raw_img = img_list[ino]
                    if raw_img is None:
                        logger.warning(f"Image at index {ino} is None. Using blank placeholder.")
                        pil_image = Image.new('RGB', (self.rec_image_shape[2], self.rec_image_shape[1]), color='white')
                        pil_image_batch.append(pil_image)
                        continue

                    if isinstance(raw_img, Image.Image):
                        pil_image = raw_img.convert('RGB')
                        pil_image_batch.append(pil_image)
                        continue

                    if not isinstance(raw_img, np.ndarray):
                        logger.error(f"Unsupported image type at index {ino}: {type(raw_img)}. Using placeholder.")
                        pil_image = Image.new('RGB', (self.rec_image_shape[2], self.rec_image_shape[1]), color='white')
                        pil_image_batch.append(pil_image)
                        continue

                    img_arr = raw_img
                    if img_arr.ndim == 2:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                    elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB)
                    elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

                    img_arr = np.asarray(img_arr)
                    if img_arr.dtype != np.uint8:
                        img_arr = np.clip(img_arr, 0, 1 if img_arr.max() <= 1.0 else 255)
                        if img_arr.max() <= 1.0:
                            img_arr = (img_arr * 255).astype(np.uint8)
                        else:
                            img_arr = img_arr.astype(np.uint8)

                    pil_image = Image.fromarray(img_arr)
                    pil_image_batch.append(pil_image)
                
                starttime = time.time()
                # Convert to PIL Images for VietOCR
                pil_images = pil_image_batch
                
                # Predict with VietOCR
                batch_results = []
                for pil_img in pil_images:
                    try:
                        text, prob = self.text_recognizer.predict(pil_img, return_prob=True)
                        if prob is None:
                            logger.success(f"VietOCR: {text}")
                            prob = 0.9
                    except Exception as e:
                        logger.error(f"VietOCR prediction failed: {str(e)}")
                        text, prob = '', 0.9
                    batch_results.append([text, prob])
                
                for rno, result in enumerate(batch_results):
                    rec_res[batch_indices[rno]] = result
                
                elapse += time.time() - starttime
                pbar.update(end_img_no - beg_img_no)

        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if isinstance(score, float) and math.isnan(score):
                rec_res[i] = (text, 0.8)
            elif score is None:
                logger.warning(f"Score is None at index {i}, setting to 0.8")
                rec_res[i] = (text, 0.8)

        return rec_res, elapse