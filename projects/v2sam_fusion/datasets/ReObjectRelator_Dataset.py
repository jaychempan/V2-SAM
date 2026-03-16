import logging
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pycocotools import mask as maskUtils
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import copy
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

SEG_QUESTIONS = [
    "Please segment the object according to the description: {class_name}",
]

SEG_QUESTIONS_SHORT = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class VideoObjectRelatorDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    PRO_CONTEXT_TOKEN = '<PRO_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                sam2_folder,
                expression_file,
                extra_image_processor=None,
                tokenizer=None,
                select_number=5,
                sampled_frames=5,
                offline_processed_text_folder=None,
                template_map_fn=None,
                max_length=8196,
                lazy=True,
                repeats=1,
                special_tokens=None,
                use_fast=False,
                n_fast_images=50,
                fast_pool_size=4,
                mode='short',
                frame_contiguous_sample=False,
    ):
        assert mode in ['long', 'long_short', 'short']
        self.mode = mode
        self.cur_mode = mode
        assert lazy is True
        # self.tokenizer = BUILDER.build(tokenizer)
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        # assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            video_ids, anno_dict = self.json_file_preprocess(expression_file)
            if self.lazy:
                self.video_ids = video_ids
                self.anno_dict = anno_dict
            else:
                raise NotImplementedError

        self.sam2_folder = sam2_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        # self._system = ''
        self._system = 'Watch the given video from another view video and labled objects.'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        print("Video res dataset (ref-sam2), include {} items.".format(len(self.video_ids)))

    def __len__(self):
        return len(self.video_ids) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.video_ids:
            cur_len = 20000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.video_ids)

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)

        video_ids = list(expression_datas.keys())
        return video_ids, expression_datas

    def dataset_map_fn(self, objects_expression_infos, prompts_expression_infos, n_frames, n_fast_frames=0):

        prompts_expressions = [object_info_['text'] for object_info_ in prompts_expression_infos]

        # prepare text
        if self.mode == 'long':
            expressions = [object_info['formated'] for object_info in objects_expression_infos]
            self.cur_mode = self.mode
        elif self.mode == 'short':
            expressions = [object_info['crop_category'] for object_info in objects_expression_infos]
            self.cur_mode = self.mode
        else:
            if random.random() < 0.5:
                expressions = [object_info['formated'] for object_info in objects_expression_infos]
                self.cur_mode = 'long'
            else:
                expressions = [object_info['crop_category'][random.randint(0, len(object_info['crop_category']) - 1)] for
                            object_info in objects_expression_infos]
                self.cur_mode = 'short'

        text_dict = self.prepare_text(n_frames, expressions, prompts_expressions, num_image_tokens=self.patch_token,
                                    n_fast_frames=n_fast_frames)
        ret = {'conversation': text_dict['conversation']}
        return ret

    def prepare_text(self, n_frames, expressions, prompts_expressions, num_image_tokens=256, n_fast_frames=0):

        if self.use_fast:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                        f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_frames * self.fast_pool_size * self.fast_pool_size}' \
                        f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{self.IMG_END_TOKEN}'

        prompt_frame_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.PRO_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{self.IMG_END_TOKEN}'
        questions = []
        prompts_questions = []
        answers = []
        for i, exp in enumerate(expressions):
            if self.cur_mode == 'short':
                question_template = random.choice(SEG_QUESTIONS_SHORT)
                prompts_question_template = prompts_expressions[i].replace(".", " another view of the video.\n")
                exp = exp.replace("a ", '')
            else:
                question_template = random.choice(SEG_QUESTIONS)
                prompts_question_template = prompts_expressions[i].replace(".", " from another view of video.\n")
            questions.append(question_template.format(class_name=exp))
            prompts_questions.append(prompts_question_template)
            # print(question_template.format(class_name=exp))
            answers.append(random.choice(ANSWER_LIST))
        qa_list = []
        for i, (prompts_question, question, answer) in enumerate(zip(prompts_questions, questions, answers)):
            #！！测试无文本的影响
            # prompts_question = 'This is the video from another view and its corresponding mask.\n'
            # question = 'Segment the corresponding object in the current views based on the mask provided for another view.'
            if i == 0:
                frame_tokens = frame_token_str + '\n'
                # frame_tokens = '=' + ' '
                frame_tokens = frame_tokens * n_frames
                frame_tokens = frame_tokens.strip()
                frame_tokens = fast_frame_token_str + frame_tokens

                prompt_frame_tokens = prompt_frame_token_str + '\n'
                # frame_tokens = '=' + ' '
                n_frames_prompt = 1
                prompt_frame_tokens = prompt_frame_tokens * n_frames_prompt
                prompt_frame_tokens = prompt_frame_tokens.strip()
                prompt_frame_tokens = fast_frame_token_str + prompt_frame_tokens
                
                qa_list.append(
                    {'from': 'human', 'value': self._system + prompt_frame_tokens + prompts_question + frame_tokens + question}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': prompts_question + question}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})

        # print(conversation)

        return {'conversation': conversation}

    def __getitem__(self, index):
        index = index % self.real_len()
        video_id = self.video_ids[index]
        expression_dict = self.anno_dict[video_id]
        object_ids = list(expression_dict['objects'].keys())

        # 处理 video_path
        if isinstance(expression_dict['video_path'], list):
            video_path = [os.path.join(self.sam2_folder, p) for p in expression_dict['video_path']]
        else:
            video_path = os.path.join(self.sam2_folder, expression_dict['video_path'])

        if isinstance(expression_dict["prompt"]['first_frame_image'], list):
            prompt_video_path = [os.path.join(self.sam2_folder, p) for p in expression_dict["prompt"]['first_frame_image']]
        else:
            prompt_video_path = os.path.join(self.sam2_folder, expression_dict["prompt"]['first_frame_image'])

        # 读取帧
        video_frames = get_video_frames(video_path)
        prompt_video_frames = get_video_frames(prompt_video_path)

        if self.use_fast:
            # sample fast branch
            fast_interval = len(video_frames) / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len(video_frames) - 1) for i in range(self.n_fast_images)]
            fast_video_frames = [video_frames[_idx] for _idx in sampled_fast_frame_idxs]
        else:
            fast_video_frames = None

        # video_frames = video_frames[::4]
        # # mask annotation
        # with open(anno_path, 'r') as f:
        #     mask_data = json.load(f)
        # masklents = decode_masklet(mask_data['masklet'])
        masklents = []
        prompt_masklents = []
        # TODO for video frames: PAN
        masklents.append(decode_segmentation_for_objects(expression_dict['objects'], object_ids))   # 每个 mask shape = (H, W, n)
        prompt_masklents.append(decode_segmentation_for_objects(expression_dict['prompt']["first_frame_anns"], object_ids))   # 每个 mask shape = (H, W, n)        
        # print(type(masklents), len(masklents))
        # print(masklents[0].shape, len(video_frames), len(masklents))

        n_frames = len(masklents)
        n_objects = len(object_ids)

        if self.select_number:
            # sample object
            if n_objects > self.select_number:
                selected_indexes = np.random.choice(n_objects, self.select_number)
            else:
                selected_indexes = np.random.choice(n_objects, self.select_number, replace=True)

            selected_object_ids = [object_ids[_idx] for _idx in selected_indexes]
            objects_expression_infos = [expression_dict['objects'][_idx] for _idx in selected_object_ids]
            prompts_expression_infos = [expression_dict['prompt']["first_frame_anns"][_idx] for _idx in selected_object_ids]

            _masklents = []
            for _mask in masklents:
                _mask_selected = []
                for _idx in selected_object_ids:
                    _mask_selected.append(_mask[:, :, int(_idx)])
                _mask_selected = np.stack(_mask_selected, axis=2)
                _masklents.append(_mask_selected)
            masklents = _masklents

            _prompt_masklents = []
            for _prompt_mask in prompt_masklents:
                _prompt_mask_selected = []
                for _idx in selected_object_ids:
                    _prompt_mask_selected.append(_prompt_mask[:, :, int(_idx)])
                _prompt_mask_selected = np.stack(_prompt_mask_selected, axis=2)
                _prompt_masklents.append(_prompt_mask_selected)
            prompt_masklents = _prompt_masklents
        else:
            objects_expression_infos = list(expression_dict['objects'].values())
            prompts_expression_infos = list(expression_dict['prompt']["first_frame_anns"].values())


        # sample video frames
        # prepare images, random select k frames
        if self.sampled_frames:
            if n_frames > self.sampled_frames + 1:
                if self.frame_contiguous_sample and random.random() < 0.5:
                    # do contiguous sample
                    selected_start_frame = np.random.choice(n_frames - self.sampled_frames, 1, replace=False)
                    selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(self.sampled_frames)]
                else:
                    selected_frame_indexes = np.random.choice(n_frames, self.sampled_frames, replace=False)
            else:
                selected_frame_indexes = np.random.choice(n_frames, self.sampled_frames, replace=True)
            selected_frame_indexes.sort()

            video_frames = [video_frames[_idx] for _idx in selected_frame_indexes]
            masklents = [masklents[_idx] for _idx in selected_frame_indexes]
            prompt_masklents = [prompt_masklents[_idx] for _idx in selected_frame_indexes]
            prompt_video_frames = [prompt_video_frames[_idx] for _idx in selected_frame_indexes]
            # ！！this is not work for understanding the video from anthor view
            # prompt_video_frames_without_masks = [prompt_video_frames[_idx] for _idx in selected_frame_indexes]
            # prompt_video_frames = add_mask2images(prompt_video_frames_without_masks, prompt_masklents)
            # add_mask2images_pair(prompt_video_frames, [prompt_masklents[0]], video_frames, [masklents[0]], save_dir='./masked_frames_with_prompt_video_frames')

        data_dict = self.dataset_map_fn(objects_expression_infos, prompts_expression_infos, len(video_frames), n_fast_frames=self.n_fast_images)


        pixel_values = []
        extra_pixel_values = []
        for frame in video_frames:
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            ori_width, ori_height = frame_image.size
            if self.extra_image_processor is not None:
                g_image = np.array(frame_image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)

            frame_image = self.transformer(frame_image)
            pixel_values.append(frame_image)

        pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
        data_dict['pixel_values'] = pixel_values
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra_pixel_values

        prompt_pixel_values = []
        prompt_extra_pixel_values = []
        for prompt_frame in prompt_video_frames:
            prompt_frame = prompt_frame[:, :, ::-1]
            prompt_frame_image = Image.fromarray(prompt_frame).convert('RGB')
            ori_width, ori_height = prompt_frame_image.size
            if self.extra_image_processor is not None:
                prompt_g_image = np.array(prompt_frame_image)  # for grounding
                prompt_g_image = self.extra_image_processor.apply_image(prompt_g_image)
                prompt_g_pixel_values = torch.from_numpy(prompt_g_image).permute(2, 0, 1).contiguous()
                prompt_extra_pixel_values.append(prompt_g_pixel_values)

            prompt_frame_image = self.transformer(prompt_frame_image)
            prompt_pixel_values.append(prompt_frame_image)

        prompt_pixel_values = torch.stack(prompt_pixel_values, dim=0)  # (n_f, 3, h, w)
        data_dict['prompt_pixel_values'] = prompt_pixel_values
        if self.extra_image_processor is not None:
            data_dict['prompt_g_pixel_values'] = prompt_extra_pixel_values

        # for fast branch
        if self.use_fast:
            fast_pixel_values = []
            for frame_image in fast_video_frames:
                frame = frame_image[:, :, ::-1]
                frame_image = Image.fromarray(frame).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values
        # print("end get item!!!!!")
        # process and get masks
        masklents = np.stack(masklents, axis=0)  # (n_frames, h, w, n_obj)
        masklents = torch.from_numpy(masklents).permute(3, 0, 1, 2)
        masklents = self._reshape_mask(masklents.flatten(0, 1))
        prompt_masklents = np.stack(prompt_masklents, axis=0)  # (n_frames, h, w, n_obj)
        prompt_masklents_tensor = torch.from_numpy(prompt_masklents).permute(3, 0, 1, 2) # (n_obj, n_frames, h, w)
        
        # wrz: 保存原始尺寸的prompt_masklents，用于sparse correspondence
        raw_prompt_masklents = prompt_masklents_tensor.clone()  # (n_obj, n_frames, h_orig, w_orig)
        
        prompt_masklents = self._reshape_mask(prompt_masklents_tensor.flatten(0, 1))  # (n_obj_total, h, w)
        
        # print('sam2-mask_shape:', masklents.shape)
        # print('sam2-pixel_values:', data_dict['pixel_values'].shape)
        # print('sam2-g_pixel_values:', len(data_dict['g_pixel_values']), ', ', data_dict['g_pixel_values'][0].shape)
        data_dict['masks'] = masklents
        data_dict['type'] = 'video'
        data_dict['prompt_masks'] = prompt_masklents
        data_dict['raw_prompt_masks'] = raw_prompt_masklents.flatten(0,1)  # wrz: 原始尺寸的prompt masks (n_obj*n_frames, h_orig, w_orig)
        vp_overall_mask = torch.Tensor([True] * len(prompt_pixel_values))
        # data_dict['vp_overall_mask'] = vp_overall_mask
        data_dict['vp_overall_mask'] = None
        
        # wrz: 添加图片原始路径，用于sparse correspondence
        # wrz: 获取第一帧的target图片路径
        # if isinstance(video_path, list):
        #     # wrz: 如果是图片列表，取第一张
        #     data_dict['target_img_path'] = video_path[0]
        # else:
        #     # wrz: 如果是单个路径，直接使用
        #     data_dict['target_img_path'] = video_path
        
        # # wrz: 获取第一帧的query(prompt)图片路径
        # if isinstance(prompt_video_path, list):
        #     # wrz: 如果是图片列表，取第一张
        #     data_dict['query_img_path'] = prompt_video_path[0]
        # else:
        #     # wrz: 如果是单个路径，直接使用
            # data_dict['query_img_path'] = prompt_video_path


        data_dict['target_img_path'] = get_video_frames_PIL(video_path)
    
        data_dict['query_img_path'] = get_video_frames_PIL(prompt_video_path)
        
        # if isinstance(video_path, list):
        #     # wrz: 如果是图片列表，取第一张
        #     data_dict['target_img_path'] = video_frames[0]
        # else:
        #     # wrz: 如果是单个路径，直接使用
        #     data_dict['target_img_path'] = video_frames
        
        # # wrz: 获取第一帧的query(prompt)图片路径
        # if isinstance(prompt_video_path, list):
        #     # wrz: 如果是图片列表，取第一张
        #     data_dict['query_img_path'] = prompt_video_frames[0]
        # else:
        #     # wrz: 如果是单个路径，直接使用
        #     data_dict['query_img_path'] = prompt_video_frames
            
        # wrz: 添加query mask路径（从prompt的annotations中获取）
        # wrz: 这里我们需要根据第一个object的segmentation生成临时mask
        # wrz: 或者直接使用已经处理好的prompt_masklents的第一个mask
        # wrz: 为了支持动态mask，我们传递None，在model中使用prompt_masks的第一帧
        data_dict['query_mask_path'] = None  # wrz: 将在forward中从prompt_masks动态生成
        
        # print("end get item!!!!!")
        
        return data_dict

    def _reshape_mask(self, masks):
        # video_masks is tensor with shape (n_obj, n_frames, h, w)
        masks = F.interpolate(
            masks.unsqueeze(0),
            size=(1024, 1024),
            mode='nearest').squeeze(0)
        return masks

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return

def get_video_frames(video_path):
    frames = []

    if isinstance(video_path, list):
        # 如果是图片路径列表
        for img_path in video_path:
            frame = cv2.imread(img_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Cannot read image {img_path}")
    else:
        # 如果是单个视频路径
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

    return frames


import imageio.v2 as imageio  # imageio.v2 兼容 imageio 旧接口

def get_video_frames_PIL(video_path):
    frames = []

    if isinstance(video_path, list):
        # 如果是图片路径列表
        for img_path in video_path:
            try:
                with Image.open(img_path) as img:
                    frame = img.convert("RGB")  # 转为RGB模式，防止有些是RGBA或L
                    frames.append(frame)
            except Exception as e:
                print(f"Warning: Cannot read image {img_path}, error: {e}")
    else:
        # 如果是单个视频路径
        try:
            reader = imageio.get_reader(video_path)
            for frame in reader:
                # 转为 PIL.Image 对象
                img = Image.fromarray(frame)
                frames.append(img)
            reader.close()
        except Exception as e:
            print(f"Error: Cannot open video file. {e}")
            return []

    return frames

def images_to_video(frames, video_name, fps=6):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()
    return

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        masks.append(mask)
    return masks

def decode_segmentation_for_objects(objects, object_ids):
    """
    解码多个对象的 segmentation，返回 (H, W, N)
    
    参数:
        objects: dict, expression_dict['objects']
        object_ids: list[str], 对象 id 列表
    
    返回:
        numpy.ndarray, (H, W, N)，N 是对象数
    """
    masks = []
    H, W = None, None

    for obj_id in object_ids:
        obj = objects[obj_id]
        if 'segmentation' in obj:
            rle = obj['segmentation']
            if isinstance(rle['counts'], list):
                rle = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
            mask = maskUtils.decode(rle).astype(np.uint8)  # (H, W)
        else:
            # 如果没有 segmentation，填充全 0
            if H is not None and W is not None:
                mask = np.zeros((H, W), dtype=np.uint8)
            else:
                raise ValueError(f"Object {obj_id} missing segmentation and size info")

        # 记录下尺寸
        if H is None or W is None:
            H, W = mask.shape
        masks.append(mask[..., None])

    return np.concatenate(masks, axis=-1)  # (H, W, N)


def draw_mask(image, mask):
    obj_mask = mask * 255
    obj_mask = np.stack([obj_mask * 1, obj_mask * 0, obj_mask * 0], axis=2)
    obj_mask = obj_mask * 0.5 + copy.deepcopy(image) * 0.5
    obj_mask = obj_mask.astype(np.uint8)
    return obj_mask

def overlay_mask(img, masks):
    """给单帧叠加绿色掩码"""
    img_copy = copy.deepcopy(img)
    if masks.ndim == 3:
        combined_mask = np.any(masks > 0, axis=-1).astype(np.uint8)
    else:
        combined_mask = (masks > 0).astype(np.uint8)
    colored_mask = np.zeros_like(img_copy)
    colored_mask[combined_mask == 1] = mask_color
    return cv2.addWeighted(img_copy, 1 - alpha, colored_mask, alpha, 0)


def add_mask2images_ori(frames, masklets):
    show_videos = []
    for i_frames, (frame, masks) in enumerate(zip(frames, masklets)):
        if i_frames == 0:
            n_obj = masks.shape[-1]
            for i_obj in range(n_obj):
                show_videos.append([])

        n_obj = masks.shape[-1]
        for i_obj in range(n_obj):
            show_videos[i_obj].append(draw_mask(copy.deepcopy(frame), masks[:, :, i_obj]))
    return show_videos


def add_mask2images_pair(frames, masklets, frames2, masklets2, 
                         save_dir='./masked_frames', alpha=0.5, mode='horizontal'):
    """
    为两组视频帧叠加绿色掩码，并将它们放在同一张图像中保存。

    参数:
        frames (list[np.ndarray]): 视频1帧列表 (H, W, 3)
        masklets (list[np.ndarray]): 视频1掩码 (H, W, n_obj)
        frames2 (list[np.ndarray]): 视频2帧列表 (H, W, 3)
        masklets2 (list[np.ndarray]): 视频2掩码 (H, W, n_obj)
        save_dir (str): 保存目录
        alpha (float): 掩码透明度 [0, 1]
        mode (str): 图像拼接方式，'horizontal'（默认）或 'vertical'

    返回:
        list[np.ndarray]: 每帧叠加掩码并组合后的图像
    """
    assert len(frames) == len(frames2), "frames 与 frames2 的长度必须相同"
    assert len(masklets) == len(masklets2), "masklets 与 masklets2 的长度必须相同"
    assert mode in ['horizontal', 'vertical'], "mode 必须为 'horizontal' 或 'vertical'"

    os.makedirs(save_dir, exist_ok=True)
    mask_color = np.array([0, 255, 0], dtype=np.uint8)  # 💚 绿色
    combined_frames = []

    for i, (f1, m1, f2, m2) in enumerate(zip(frames, masklets, frames2, masklets2)):
        blended1 = overlay_mask(f1, m1)
        blended2 = overlay_mask(f2, m2)

        # 调整大小以匹配（避免不同尺寸导致拼接错误）
        h1, w1 = blended1.shape[:2]
        h2, w2 = blended2.shape[:2]
        if (h1, w1) != (h2, w2):
            new_w, new_h = min(w1, w2), min(h1, h2)
            blended1 = cv2.resize(blended1, (new_w, new_h))
            blended2 = cv2.resize(blended2, (new_w, new_h))

        # 拼接方式
        if mode == 'horizontal':
            combined = np.concatenate((blended1, blended2), axis=1)
        else:  # vertical
            combined = np.concatenate((blended1, blended2), axis=0)

        combined_frames.append(combined)

        # 保存结果
        save_path = os.path.join(save_dir, f"combined_{i:04d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    return combined_frames


def add_mask2images(frames, masklets, save_dir='./masked_frames', alpha=0.5):
    """
    为每帧叠加所有对象的绿色掩码，并可选择保存结果。

    参数:
        frames (list[np.ndarray]): 视频帧列表 (H, W, 3)
        masklets (list[np.ndarray]): 每帧对应的掩码 (H, W, n_obj)
        save_dir (str): 若指定，则保存结果到该目录
        alpha (float): 掩码透明度，默认 0.5

    返回:
        list[np.ndarray]: 每帧叠加掩码后的新帧列表
    """
    frames_with_masks = []
    mask_color = np.array([0, 255, 0], dtype=np.uint8)  # 💚 绿色掩码

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i_frame, (frame, masks) in enumerate(zip(frames, masklets)):
        img = copy.deepcopy(frame)
        n_obj = masks.shape[-1]

        # 合并所有对象的掩码
        combined_mask = np.zeros(masks.shape[:2], dtype=np.uint8)
        for i_obj in range(n_obj):
            combined_mask = np.logical_or(combined_mask, masks[:, :, i_obj] > 0)

        combined_mask = combined_mask.astype(np.uint8)

        # 创建彩色掩码层
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[combined_mask == 1] = mask_color

        # 混合叠加
        blended = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
        frames_with_masks.append(blended)

        # 保存帧
        if save_dir:
            save_path = os.path.join(save_dir, f"frame_{i_frame:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    return frames_with_masks
