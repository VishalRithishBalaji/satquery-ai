from __future__ import annotations
from threading import Lock
from typing import Any
import torch
from PIL import Image

class GeoChatModel:
    _instance = None; _lock = Lock()
    def __init__(self, model_id: str, base_model_id: str, max_pixels: int = 401408):
        self.model_id=model_id; self.base_model_id=base_model_id; self.max_pixels=max_pixels
        self.device='cuda' if torch.cuda.is_available() else 'cpu'; self.model=None; self.processor=None
        self.loaded=False; self.load_error=None
    @classmethod
    def singleton(cls, model_id, base_model_id, max_pixels):
        with cls._lock:
            if cls._instance is None or cls._instance.model_id!=model_id or cls._instance.base_model_id!=base_model_id:
                cls._instance=cls(model_id,base_model_id,max_pixels)
            return cls._instance
    def load(self):
        if self.loaded: return self
        try:
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            dtype=torch.float16 if self.device=='cuda' else torch.float32
            kwargs={'torch_dtype':dtype}
            if self.device=='cuda': kwargs['device_map']='auto'
            self.model=Qwen2VLForConditionalGeneration.from_pretrained(self.base_model_id, **kwargs)
            self.model=PeftModel.from_pretrained(self.model,self.model_id)
            self.model.eval()
            self.processor=AutoProcessor.from_pretrained(self.base_model_id,min_pixels=224*224,max_pixels=self.max_pixels)
            self.loaded=True; self.load_error=None; return self
        except Exception as e:
            self.load_error=f'{type(e).__name__}: {e}'; self.unload(); raise
    def unload(self):
        self.model=None; self.processor=None; self.loaded=False
        if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        return self

    def _generate_plain(self, images: list[Image.Image], prompt: str, max_new_tokens=128, temperature=0.15):
        self.load()
        from qwen_vl_utils import process_vision_info
        content=[{'type':'image','image':im} for im in images]; content.append({'type':'text','text':prompt})
        messages=[{'role':'user','content':content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        image_inputs,video_inputs=process_vision_info(messages)
        inputs=self.processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt')
        if self.device=='cuda': inputs=inputs.to('cuda')
        with torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=temperature>0,temperature=temperature,use_cache=True)
        generated=[out[len(inp):] for inp,out in zip(inputs.input_ids,ids)]
        return self.processor.batch_decode(generated,skip_special_tokens=True)[0].strip()

    def _generate_with_certainty(self, images: list[Image.Image], prompt: str, max_new_tokens=128, temperature=0.15):
        """Generate and additionally estimate mean per-token model certainty
        from the generation logits. Falls back to the plain generation path
        (with a neutral certainty) if score extraction is not supported by
        the installed transformers/generation configuration, so an optional
        confidence signal can never break analysis."""
        self.load()
        from qwen_vl_utils import process_vision_info
        content=[{'type':'image','image':im} for im in images]; content.append({'type':'text','text':prompt})
        messages=[{'role':'user','content':content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        image_inputs,video_inputs=process_vision_info(messages)
        inputs=self.processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt')
        if self.device=='cuda': inputs=inputs.to('cuda')
        try:
            with torch.inference_mode():
                output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=temperature>0,temperature=temperature,use_cache=True,output_scores=True,return_dict_in_generate=True)
            ids=output.sequences
            generated=[out[len(inp):] for inp,out in zip(inputs.input_ids,ids)]
            decoded=self.processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            certainty=self._mean_token_confidence(output.scores,generated[0])
            return decoded, certainty
        except Exception:
            decoded=self._generate_plain(images,prompt,max_new_tokens=max_new_tokens,temperature=temperature)
            return decoded, 0.5

    @staticmethod
    def _mean_token_confidence(scores, generated_ids) -> float:
        try:
            if not scores or generated_ids is None or len(generated_ids)==0: return 0.5
            probs=[]
            for step_logits, token_id in zip(scores, generated_ids):
                step_probs=torch.softmax(step_logits[0].float(), dim=-1)
                probs.append(step_probs[int(token_id)].item())
            if not probs: return 0.5
            return max(0.0, min(1.0, sum(probs)/len(probs)))
        except Exception:
            return 0.5

    def ask(self,image,prompt,**kwargs): return self._generate_plain([image],prompt,**kwargs)
    def ask_multiple(self,images,prompt,**kwargs): return self._generate_plain(images,prompt,**kwargs)
    def ask_with_certainty(self,image,prompt,**kwargs): return self._generate_with_certainty([image],prompt,**kwargs)
    def ask_multiple_with_certainty(self,images,prompt,**kwargs): return self._generate_with_certainty(images,prompt,**kwargs)

    def info(self):
        return {'loaded':self.loaded,'model_id':self.model_id,'base_model_id':self.base_model_id,'device':self.device,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'error':self.load_error}
