import os
import io
import re
import base64
import hashlib
import argparse
import unicodedata
from tqdm import tqdm
from typing import List, Dict
from multiprocessing.pool import ThreadPool
from PIL import Image

import fitz

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import (
    post_process_output,
    pre_process_bboxes,
)
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    parse image or pdf file
    """

    def __init__(
        self,
        ip="localhost",
        port=8000,
        model_name="model",
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=16384,
        num_thread=64,
        dpi=200,
        min_pixels=None,
        max_pixels=None,
        use_hf=False,
    ):
        self.dpi = dpi

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        if self.use_hf:
            self._load_hf_model()
            print("use hf model, num_thread will be set to 1")
        else:
            print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_path = "./weights/DotsOCR"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.process_vision_info = process_vision_info

    def _inference_with_hf(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt,
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(
        self,
        prompt_mode,
        bbox=None,
        origin_image=None,
        image=None,
        min_pixels=None,
        max_pixels=None,
    ):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == "prompt_grounding_ocr":
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(
                origin_image,
                bboxes,
                input_width=image.width,
                input_height=image.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
    ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None:
            assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None:
            assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == "image" and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(
                origin_image, min_pixels=min_pixels, max_pixels=max_pixels
            )
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(
            prompt_mode,
            bbox,
            origin_image,
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {
            "page_idx": page_idx,
            "input_height": input_height,
            "input_width": input_width,
            "image": image,
        }
        if prompt_mode in [
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_grounding_ocr",
        ]:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if (
                filtered and prompt_mode != "prompt_layout_only_en"
            ):  # model output json failed, use filtered process
                pass
            else:
                # dd bbox info and croped image to result
                for i, cell in enumerate(cells):
                    bbox = cell["bbox"]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:  # invalid bbox
                        continue
                    cell["image"] = origin_image.crop(
                        (bbox[0], bbox[1], bbox[2], bbox[3])
                    )
                result.update({"cells": cells})

                if (
                    prompt_mode != "prompt_layout_only_en"
                ):  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key="text")
                    result.update({"md_content": md_content})

        else:
            md_content = response
            result.update({"md_content": md_content})

        return result

    def parse_image(self, input_path, prompt_mode, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(
            origin_image,
            prompt_mode,
            source="image",
            bbox=bbox,
            fitz_preprocess=fitz_preprocess,
        )
        # result['file_path'] = input_path
        return [result]

    def parse_pdf(self, input_path, prompt_mode):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "source": "pdf",
                "page_idx": i,
            }
            for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        if self.use_hf:
            num_thread = 1
        else:
            num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_idx"])
        # for i in range(len(results)):
        #     results[i]['file_path'] = input_path
        return results

    def parse_file(
        self,
        input_path,
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
    ):

        filename, file_ext = os.path.splitext(os.path.basename(input_path))


        if file_ext == ".pdf":
            results = self.parse_pdf(input_path, prompt_mode)
        elif file_ext in image_extensions:
            results = self.parse_image(
                input_path, prompt_mode, bbox=bbox, fitz_preprocess=fitz_preprocess
            )
        else:
            raise ValueError(
                f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf"
            )

        return results

    def batch_parse_images(self):
        pass


# class DotsOCRLayoutParser:
#     def __init__(self):
#         pass

#     def __call__(self, pdf_path: str):

    


class DotsOCRImageInfoParser:
    """
    从 PDF 文件中提取图像及其对应的图注
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.fitz_doc = fitz.open(pdf_path)
        self.layout_info = []
        # self.layout_info example:
        # [
        #     {"page_idx": 0, "cells": [{"category": "Picture", "bbox": [1, 2, 3, 4], "image": Image}, ...]},
        #     ...
        # ]
        self.DPI = 300
        self.dots_ocr_parser = DotsOCRParser(
            ip="localhost",
            port=8000,
            model_name="model",
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=16,
            dpi=self.DPI,
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
        )
        self.fitz_preprocess = True
    
    def __call__(self):
        self.layout_parse()
        self.extract_captions()
        self.extract_image_captions()
        return self.get_result()

    def get_result(self):
        image_info = []
        for page_result in self.layout_info:
            page_idx = page_result.get("page_idx", 0)
            cells = page_result.get("cells", [])
            for cell in cells:
                if cell.get("category") == "Picture":
                    image = cell.get("image", None)
                    caption = cell.get("caption", "")
                    if image is None:
                        continue
                    image_ext = image.format.lower() if image.format else "png"
                    buffer = io.BytesIO()
                    image.save(buffer, format=image_ext.upper())
                    image_bytes = buffer.getvalue()
                    image_md5 = hashlib.md5(image.tobytes()).hexdigest()
                    safe_image_name = self.safe_filename(f"page_{page_idx+1}_{image_md5}.{image_ext}")
                    image_info.append(
                        {
                            "page_idx": page_idx + 1,
                            "image_name": safe_image_name,
                            "image_md5": image_md5,
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                            "caption": caption if caption else "",
                        }
                    )
        return image_info

    def safe_filename(self, name: str) -> str:
        # 把不允许的字符替换成下划线
        return re.sub(r'[\/:*?"<>|]', "_", name)
    
    def extract_captions(self):
        """
        根据layout_parse的结果，提取图注
        1. 先用fitz提取
        2. 如果fitz提取的文本为空，则用OCR提取
        """
        for page_idx in range(len(self.layout_info)):
            result = self.layout_info[page_idx]
            cells = result.get("cells", [])
            for idx in range(len(cells)):
                cell = cells[idx]
                if cell.get("category") == "Caption":
                    cells[idx] = self.get_image_caption_with_fitz(page_idx, cell)
        return self
    
    def get_image_caption_with_ocr(self, page_idx, caption_cell):
        """
        根据caption的bbox，使用ocr大模型提取对应区域的文字
        """
        caption_image = caption_cell.get("image", None)
        if caption_image is None:
            return caption_cell

        output = self.dots_ocr_parser.parse_image(
            caption_image,
            prompt_mode="prompt_ocr",
            bbox=None,
            fitz_preprocess=self.fitz_preprocess,
        )
        text = output[0].get("md_content", "").strip()
        caption_cell["text"] = self.clean_captions(text)
        return caption_cell

    def get_image_caption_with_fitz(self, page_idx, caption_cell):
        """
        根据caption的bbox，使用fitz提取对应区域的文字
        """
        caption_bbox = caption_cell["bbox"]
        page = self.fitz_doc[page_idx]
        scale = self.DPI / 72
        x0, y0, x1, y1 = [coord / scale for coord in caption_bbox]
        text_bbox = (x0, y0, x1, y1)
        rect = fitz.Rect(text_bbox)
        text = page.get_text("text", clip=rect).strip()
        text = self.clean_captions(text, text_bbox)
        caption_cell["text"] = text
        # 如果文本为空，需要走OCR
        if text == "":
            caption_cell = self.get_image_caption_with_ocr(page_idx, caption_cell)
        return caption_cell
    
    def layout_parse(self):
        self.layout_info = self.dots_ocr_parser.parse_file(
            self.pdf_path,
            prompt_mode="prompt_layout_only_en",
            bbox=None,
            fitz_preprocess=self.fitz_preprocess,
        )
        return self

    def extract_image_captions(self):
        """
        从results中提取图像及其对应的图注
        """
        for result in self.layout_info:
            if "cells" in result:
                cells = result["cells"]
                self.extract_single_page_image_captions(cells)
        return self

    def extract_single_page_image_captions(self, cells):
        """
        从cells中提取图像及其对应的图注，更新self.layout_info
        """
        image_cells = [cell for cell in cells if cell.get("category") == "Picture"]
        caption_cells = [cell for cell in cells if cell.get("category") == "Caption"]
        for image_cell in image_cells:
            caption = self.extract_single_image_caption(image_cell, caption_cells)
            if caption:
                image_cell["caption"] = caption

    def extract_single_image_caption(
        self, image_cell: Dict, caption_cells: List[Dict]
    ) -> str:
        image = image_cell.get("image", None)
        if image is None:
            return None
        image_bbox = image_cell["bbox"]
        image_left, image_top, image_right, image_bottom = image_bbox
        image_width = image_right - image_left
        image_height = image_bottom - image_top
        if image_width <= 50 or image_height <= 50:
            # 过滤掉过小的图片
            return None
        if min(image_width, image_height) / max(image_width, image_height) < 0.2:
            # 过滤掉过于狭长的图片
            return None
        caption_candidates = []
        for caption_cell in caption_cells:
            caption = caption_cell.get("text", None)
            if caption is None:
                continue
            caption_bbox = caption_cell["bbox"]
            distance, relation = self._rect_metrics(image_bbox, caption_bbox)
            # 方向权重
            distance_weight = {
                "below_center": 1.1,
                "below_offside": 1.2,
                "overlap": 1.2,
                "left": 1.2,
                "right": 1.2,
                "above_center": 1.1,
                "above_offside": 2,
            }.get(relation, 10)

            # 打分
            if relation in ["left", "right"]:
                distance = distance / (image_width / 2)  # 水平距离相对于图像宽度归一化
            else:
                distance = distance / (image_height / 2)  # 垂直距离相对于图像高度归一化
            score = distance * distance_weight

            caption_candidates.append(
                {
                    "caption": caption,
                    "bbox": caption_bbox,
                    "relation": relation,
                    "score": score,
                }
            )
        # 如果没有找到任何图注，返回空字符串
        if not caption_candidates:
            return None
        # 按距离评分排序，选择最合适的图注
        caption_candidates.sort(key=lambda x: x["score"])
        # 只返回距离图像较近的图注（在合理范围内）
        closest_caption = caption_candidates[0]
        # 设置距离阈值（页面高度的25%）
        distance_threshold = 1.3
        if closest_caption["score"] <= distance_threshold:
            # 生成坐标信息字符串
            # coord_info = f"position: {closest_caption['relation']})"
            return closest_caption["caption"]
        else:
            return None

    def _rect_metrics(self, image_bbox, text_bbox):
        """
        计算图像bbox和文本bbox的距离和方向关系
        """
        image_left, image_top, image_right, image_bottom = image_bbox
        text_left, text_top, text_right, text_bottom = text_bbox

        text_center_x = (text_left + text_right) / 2
        text_center_y = (text_top + text_bottom) / 2

        image_width = image_right - image_left
        image_height = image_bottom - image_top

        distance = float("inf")

        # ====== 方向关系 ======
        relation = "other"
        if text_top >= image_bottom:  # 只保留 below
            # relation = "below"
            if text_center_x >= image_left and text_center_x <= image_right:
                relation = "below_center"
                distance = text_top - image_bottom + image_height / 2
            else:
                relation = "below_offside"
                distance = (
                    text_top
                    - image_bottom
                    + image_height / 2
                    + min(
                        abs(text_center_x - image_left),
                        abs(text_center_x - image_right),
                    )
                )
        elif text_bottom <= image_top:
            if text_center_x >= image_left and text_center_x <= image_right:
                relation = "above_center"
                distance = image_top - text_bottom + image_height / 2
            else:
                relation = "above_offside"
                distance = (
                    image_top
                    - text_bottom
                    + image_height / 2
                    + min(
                        abs(text_center_x - image_left),
                        abs(text_center_x - image_right),
                    )
                )
        elif text_right <= image_left:
            relation = "left"
            distance = image_left - text_right + image_width / 2
        elif text_left >= image_right:
            relation = "right"
            distance = text_left - image_right + image_width / 2
        elif (
            text_right > image_left
            and text_left < image_right
            and text_bottom > image_top
            and text_top < image_bottom
        ):
            relation = "overlap"
            distance = min(
                abs(text_center_x - image_left),
                abs(image_right - text_center_x),
                abs(text_center_y - image_top),
                abs(image_bottom - text_center_y),
            )

        return distance, relation

    def clean_captions(self, text: str, bbox=None) -> str:
        """
        清理从 PDF 提取的图注：
        1. 把各种 Unicode 空格（全角空格、em space、en space 等）统一为半角空格
        2. 去掉零宽字符（如 \u200b, \u200c, \u200d）
        3. 去掉控制符（如 \u2028, \u2029）
        4. 合并连续空格为一个
        5. 去掉私有区符号（\ue000-\uf8ff）
        6. 规范化“图”字与编号之间的空格
        7. 根据 bbox 判断横排或竖排，竖排去换行拼接，横排去多余空格
        """
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat == "Zs":
                cleaned.append(" ")
            elif ch in ["\u200b", "\u200c", "\u200d", "\ufeff"]:
                continue
            elif ch in ["\u2028", "\u2029"]:
                cleaned.append("\n")
            else:
                cleaned.append(ch)

        text = "".join(cleaned)
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r'[\ue000-\uf8ff]', '', text)
        text = re.sub(r'图\s*([一二三四五六七八九十\d]+)', r'图\1', text)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            width = x1 - x0
            height = y1 - y0
            if height > width * 4:  # 竖排
                return text.replace("\n", "").replace(" ", "")
            else:  # 横排
                return " ".join(text.split())
        return text.strip()

    def __del__(self):
        """清理资源，防止内存泄漏"""
        if hasattr(self, 'fitz_doc') and self.fitz_doc:
            self.fitz_doc.close()
        # 清理大型对象引用
        self.layout_info = []


def main():
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    parser.add_argument("input_path", type=str, help="Input PDF/image file path")

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    args = parser.parse_args()
    DPI = 300

    dots_ocr_parser = DotsOCRParser(
        ip="localhost",
        port=8000,
        model_name="model",
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=16384,
        num_thread=16,
        dpi=DPI,
        output_dir="./output",
        min_pixels=None,
        max_pixels=None,
        use_hf=False,
    )

    fitz_preprocess = True
    if fitz_preprocess:
        print(
            "Using fitz preprocess for image input, check the change of the image pixels"
        )
    results = dots_ocr_parser.parse_file(
        args.input_path,
        prompt_mode="prompt_layout_only_en",
        bbox=None,
        fitz_preprocess=fitz_preprocess,
    )

    # post process to keep only picture and caption for easy visualization
    for result in results:
        if "cells" in result:
            result["cells"] = [
                {
                    "page_idx": result["page_idx"],
                    "category": cell.get("category"),
                    "bbox": cell.get("bbox"),
                    "image": cell.get("image"),
                }
                for cell in result["cells"]
                if cell.get("category") in ["Caption"]
            ]
        else:
            result["cells"] = []
    
    
    doc = fitz.open(args.input_path)
    scale = DPI / 72

    for page_index in range(len(doc)):
        page = doc[page_index]
        result = results[page_index]
        for cell in result["cells"]:
            bbox = cell.get("bbox")
            x0, y0, x1, y1 = [coord / scale for coord in bbox]
            text_bbox = (x0, y0, x1, y1)
            rect = fitz.Rect(x0, y0, x1, y1)
            text = page.get_text("text", clip=rect)
            # TODO 如果文本为空，需要走OCR
            area = (x1 - x0) * (y1 - y0)
            if area > 10000:
                print("*" * 100)
                print(f"Page {page_index+1} Caption: {text}, x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}, 面积：{area}")
                print("*" * 100)

                continue
            # print(f"Page {page_index+1} Caption: {clean_caption(text, text_bbox)}, x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}, 面积：{area}")

if __name__ == "__main__":
    # main()
    import sys
    pdf_path = sys.argv[1]
    parser = DotsOCRImageInfoParser(pdf_path)
    results = parser()
    import json
    # save to file
    with open("image_captions.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
