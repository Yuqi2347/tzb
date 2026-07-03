import math
from typing import List, Optional
import json
import torch
import torchvision

from threading import Thread
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .configuration_fm9g import FM9GVConfig
from .modeling_fm9g import FM9GPreTrainedModel, FM9GForCausalLM
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler

from transformers import SiglipVisionConfig

class FM9GVPreTrainedModel(FM9GPreTrainedModel):
    config_class = FM9GVConfig


class FM9GV(FM9GVPreTrainedModel):
    _no_split_modules = ["SiglipEncoderLayer", "Mona"]
    def __init__(self, config):
        super().__init__(config)
        self.llm = FM9GForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '</s>']

    def init_vision_module(self):

        # if isinstance(self.config.vision_config, dict):
        #     self.config.vision_config = SiglipVisionConfig(**self.config.vision_config)
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=vision_dim // 72,
            kv_dim=vision_dim,
            adaptive=True
        )
    # def _init_weights(self, module):
    #     if hasattr(module, 'mona'):
    #         return
    #     super()._init_weights(module)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import torch
        import os
        from transformers.utils import is_safetensors_available

        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)

        # 检查权重文件
        model_path = pretrained_model_name_or_path
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            if is_safetensors_available():
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
            else:
                raise ImportError("safetensors not available")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            # 处理分片模型
            import glob
            safetensor_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
            if safetensor_files:
                if is_safetensors_available():
                    from safetensors.torch import load_file
                    state_dict = {}
                    for file in safetensor_files:
                        state_dict.update(load_file(file))
                else:
                    raise ImportError("safetensors not available")
            else:
                bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
                if not bin_files:
                    bin_files = glob.glob(os.path.join(model_path, "model-*.bin"))
                
                if bin_files:
                    state_dict = {}
                    for file in sorted(bin_files):
                        state_dict.update(torch.load(file, map_location='cpu'))
                else:
                    raise FileNotFoundError(f"No model weights found in {model_path}")

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("⚠️ 缺失的权重（missing_keys）:", missing_keys)
        print("❗ 未使用的权重（unexpected_keys）:", unexpected_keys)
        # 检查 Mona 权重加载情况
        mona_keys = [k for k in state_dict.keys() if 'mona1' in k.lower() or 'mona2' in k.lower()]
        mona_missing = [k for k in missing_keys if 'mona1' in k.lower() or 'mona2' in k.lower()]

        if mona_keys:
            loaded_mona = len(mona_keys) - len(mona_missing)
            print(f"Mona权重加载: {loaded_mona}/{len(mona_keys)} 个参数")
            if mona_missing:
                print(f"未加载的Mona参数: {mona_missing[:3]}..." if len(mona_missing) > 3 else mona_missing)

        # 处理 torch_dtype
        torch_dtype = kwargs.get('torch_dtype', None)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)

                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]

                # 调试信息
                # print(f"[DEBUG] Batch {i}:")
                # print(f"  cur_vllm_emb.shape: {cur_vllm_emb.shape}")
                # print(f"  cur_vs_hs.shape: {cur_vs_hs.shape}")
                # print(f"  cur_image_bound: {cur_image_bound}")
                # print(f"  Number of images in bound: {len(cur_image_bound)}")

                if len(cur_image_bound) > 0:
                    # 检查vision tokens和image_bound的匹配性
                    num_images = len(cur_image_bound)
                    if cur_vs_hs.dim() == 3:  # [num_images, tokens_per_image, hidden_size]
                        available_images = cur_vs_hs.shape[0]
                        tokens_per_image = cur_vs_hs.shape[1]
                    elif cur_vs_hs.dim() == 2:  # [total_tokens, hidden_size]
                        total_tokens = cur_vs_hs.shape[0]
                        # 假设每个图像的token数相同
                        tokens_per_image = total_tokens // num_images if num_images > 0 else total_tokens
                        available_images = num_images
                    else:
                        raise ValueError(f"Unexpected cur_vs_hs dimensions: {cur_vs_hs.shape}")

                    # print(f"  Available images: {available_images}, Required images: {num_images}")
                    # print(f"  Tokens per image: {tokens_per_image}")

                    # 处理图像数量不匹配的情况
                    if available_images < num_images:
                        # print(f"[WARNING] Not enough vision features for all images")
                        # print(f"  Available: {available_images}, Required: {num_images}")

                        # 策略1: 重复使用可用的vision features
                        if available_images == 1 and cur_vs_hs.dim() == 3:
                            # 重复单个图像的features来匹配所有图像位置
                            repeated_features = cur_vs_hs[0:1].repeat(num_images, 1,
                                                                      1)  # [num_images, tokens_per_image, hidden_size]
                            cur_vs_hs = repeated_features
                            # print(f"[FIX] Repeated single image features to shape: {cur_vs_hs.shape}")

                        elif available_images == 1 and cur_vs_hs.dim() == 2:
                            # 重复单个图像的features
                            single_image_tokens = cur_vs_hs  # [tokens_per_image, hidden_size]
                            repeated_features = single_image_tokens.unsqueeze(0).repeat(num_images, 1,
                                                                                        1)  # [num_images, tokens_per_image, hidden_size]
                            cur_vs_hs = repeated_features
                            # print(f"[FIX] Created repeated features with shape: {cur_vs_hs.shape}")

                        # 策略2: 截断image_bound到可用的图像数量
                        else:
                            cur_image_bound = cur_image_bound[:available_images]
                            # print(f"[FIX] Truncated image_bound to: {cur_image_bound}")

                    # 执行scatter操作
                    try:
                        image_indices = torch.stack(
                            [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                        ).to(vllm_embedding.device)

                        # print(f"  Final image_indices.shape: {image_indices.shape}")
                        # print(f"  Final cur_vs_hs.shape: {cur_vs_hs.shape}")

                        # 确保形状匹配
                        if cur_vs_hs.dim() == 3:
                            vs_tokens = cur_vs_hs.view(-1, cur_vs_hs.shape[-1])  # [total_tokens, hidden_size]
                        else:
                            vs_tokens = cur_vs_hs

                        # 最终安全检查
                        expected_tokens = image_indices.numel()
                        available_tokens = vs_tokens.shape[0]

                        if expected_tokens != available_tokens:
                            # print(f"[WARNING] Token count mismatch: expected {expected_tokens}, available {available_tokens}")
                            min_tokens = min(expected_tokens, available_tokens)
                            image_indices = image_indices.view(-1)[:min_tokens].view(-1, 1)
                            vs_tokens = vs_tokens[:min_tokens]
                            # print(f"[FIX] Adjusted to use {min_tokens} tokens")

                        cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                              vs_tokens.view(-1, vs_tokens.shape[-1]))

                        # print(f"[SUCCESS] Scatter operation completed")

                    except RuntimeError as e:
                        print(f"[ERROR] Scatter operation still failed: {e}")
                        # 最后的备用方案：手动赋值
                        flat_indices = image_indices.view(-1)
                        flat_vs_hs = vs_tokens.view(-1, vs_tokens.shape[-1])

                        for idx, vs_token in zip(flat_indices, flat_vs_hs):
                            if 0 <= idx < cur_vllm_emb.shape[0]:
                                cur_vllm_emb[idx] = vs_token
                        print(f"[FALLBACK] Used manual assignment")

                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0
        # bs = len(data['input_ids'])
        # for i in range(bs):
        #     cur_vs_hs = vision_hidden_states[i]
        #     if len(cur_vs_hs) > 0:
        #         cur_vllm_emb = vllm_embedding[i]
        #         cur_image_bound = data['image_bound'][i]
        #
        #         # 添加调试信息
        #         # print(f"[DEBUG] Batch {i}:")
        #         # print(f"  cur_vllm_emb.shape: {cur_vllm_emb.shape}")
        #         # print(f"  cur_vs_hs.shape: {cur_vs_hs.shape}")
        #         # print(f"  cur_image_bound: {cur_image_bound}")
        #         # print(f"  input_ids length: {len(data['input_ids'][i])}")
        #
        #         if len(cur_image_bound) > 0:
        #             image_indices = torch.stack(
        #                 [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
        #             ).to(vllm_embedding.device)
        #
        #             cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
        #                                   cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        #         elif self.training:
        #             cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )
    
    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()
    
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, decode_text=decode_text, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states
        
        return result

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192, #8192
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image
        
        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor
        
        assert self.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs        

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt", 
            max_length=max_inp_length
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
            
        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config
            )
        
        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer
