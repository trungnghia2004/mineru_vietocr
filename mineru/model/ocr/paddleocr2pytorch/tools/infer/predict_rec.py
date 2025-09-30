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
from transformers import pipeline

class TextRecognizer:
    def __init__(self, args, **kwargs):
        # Load VietOCR configuration
        self.device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        vietocr_config = Cfg.load_config_from_name('vgg_transformer')
        vietocr_config['device'] = self.device
        vietocr_config['cnn']['pretrained'] = True
        vietocr_config['predictor']['beamsearch'] = True
        self.text_recognizer = Predictor(vietocr_config)
        
        # Batch processing parameters
        self.rec_batch_num = args.rec_batch_num if hasattr(args, 'rec_batch_num') else 6
        self.rec_image_shape = [3, 32, 320]  # Default VietOCR input shape: [channels, height, width]
        self.limited_max_width = args.limited_max_width if hasattr(args, 'limited_max_width') else 1280
        self.limited_min_width = args.limited_min_width if hasattr(args, 'limited_min_width') else 16
        self.expand_right_ratio = args.expand_right_ratio if hasattr(args, 'expand_right_ratio') else 0.1  # 10% width
        
        # Load text correction pipeline
        try:
            self.corrector = pipeline(
                "text2text-generation",
                model="bmd1905/vietnamese-correction-v2",
                device=0 if self.device == 'cuda' else -1  # Use GPU if available
            )
            self.max_length = args.max_length if hasattr(args, 'max_length') else 128
            logger.info("Vietnamese correction pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load text correction pipeline: {str(e)}")
            self.corrector = None

    def resize_norm_img(self, img):
        """Resize and normalize image for VietOCR input, with right-side expansion."""
        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape[:2]
        
        # Skip invalid images
        if h == 0 or w == 0:
            logger.warning(f"Invalid image dimensions: height={h}, width={w}")
            return None
        
        # Expand right side of the image
        expand_pixels = int(w * self.expand_right_ratio)  # Expand by 10% of width
        logger.debug(f"Expanding right side by {expand_pixels} pixels")
        expanded_w = w + expand_pixels
        
        # Create new image with expanded width
        expanded_img = np.zeros((h, expanded_w, imgC), dtype=img.dtype)
        expanded_img[:, :w, :] = img  # Copy original image to left side
        
        # Calculate aspect ratio and resize
        ratio = expanded_w / float(h)
        resized_w = int(imgH * ratio)
        resized_w = max(min(resized_w, self.limited_max_width), self.limited_min_width)
        
        # Resize image
        resized_image = cv2.resize(expanded_img, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)
        resized_image = resized_image.astype('float32')
        
        # Convert to RGB and normalize
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image / 255.0  # VietOCR expects [0, 1] range
        
        # Pad image if necessary
        padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        if resized_w < imgW:
            padding_im[:, :resized_w, :] = resized_image
        else:
            padding_im = resized_image[:, :imgW, :]
        
        logger.debug(f"Processed image shape: {padding_im.shape}")
        return padding_im

    def correct_with_pipeline(self, texts):
        """Correct spelling errors in a list of texts using the text2text-generation pipeline."""
        if self.corrector is None:
            logger.warning("Text correction pipeline not loaded. Returning original texts.")
            return texts
        
        if not texts or all(not text.strip() for text in texts):
            logger.warning("Input texts are empty or all blank. Skipping correction.")
            return texts
        
        try:
            # Use the pipeline to correct texts
            predictions = self.corrector(texts, max_length=self.max_length)
            corrected_texts = [pred['generated_text'] for pred in predictions]
            return corrected_texts
        except Exception as e:
            logger.error(f"Error in text correction pipeline: {str(e)}")
            return texts

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict"):
        """Process a list of images with VietOCR and correct spelling with the pipeline."""
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
                norm_img_batch = []
                batch_indices = indices[beg_img_no:end_img_no]
                
                # Preprocess images for the batch
                for ino in batch_indices:
                    norm_img = self.resize_norm_img(img_list[ino])
                    if norm_img is None:
                        norm_img_batch.append(np.zeros((32, 320, 3), dtype=np.float32))
                        continue
                    norm_img_batch.append(norm_img)
                
                norm_img_batch = np.array(norm_img_batch)  # Shape: [batch, H, W, C]
                
                starttime = time.time()
                # Convert to PIL Images for VietOCR
                pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in norm_img_batch]
                
                # Predict with VietOCR
                batch_results = []
                batch_texts = []
                for pil_img in pil_images:
                    try:
                        text, prob = self.text_recognizer.predict(pil_img, return_prob=True)
                        if prob is None:
                            logger.success(f"VietOCR: {text}")
                            prob = 0.9
                        batch_texts.append(text)
                    except Exception as e:
                        logger.error(f"VietOCR prediction failed: {str(e)}")
                        text, prob = '', 0.9
                    batch_results.append([text, prob])
                
                corrected_texts = self.correct_with_pipeline(batch_texts)
                logger.success(f"Correction: {corrected_texts}")
                # Assign corrected results back to original indices
                for rno, (corrected_text, orig_result) in enumerate(zip(corrected_texts, batch_results)):
                    batch_results[rno] = [corrected_text, orig_result[1]]  # Keep original score
                    rec_res[batch_indices[rno]] = batch_results[rno]
                
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