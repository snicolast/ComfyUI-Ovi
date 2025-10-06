
import torch
import torch.nn as nn
from ovi.modules.model import WanLayerNorm, WanModel, WanRMSNorm, gradient_checkpointing, rope_apply
from ovi.modules.attention import flash_attention
from ovi.distributed_comms.communications import all_gather, all_to_all_4D
from ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state

class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        has_video = True 
        has_audio = True
        if video_config is not None:
            self.video_model = WanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            self.audio_model = WanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.use_sp = get_sequence_parallel_state()
            if self.use_sp:
                self.sp_size = nccl_info.sp_size
                self.sp_rank = nccl_info.rank_within_group
            self.inject_cross_attention_kv_projections()

        self.init_weights()
        
    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()

        
        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()


    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_cross_attention_forward(self,
                                            cross_attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens
                                            ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            ## means is i2v block
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            ## means is t2v block
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

        
        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            if k_img is not None:
                k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            if v_img is not None:
                v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        if self.use_sp: 
            k_target = all_to_all_4D(k_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
            v_target = all_to_all_4D(v_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x = x + target_x
        if self.use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        
        x = x.flatten(2) # [B, L/P, C]

        x = cross_attn_block.o(x)
        return x

    def single_fusion_cross_attention_ffn_forward(self,
                                            attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens,
                                            src_e):
        
        src_seq = src_seq + self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens
                                                                       )
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq
        
    def single_fusion_block_forward(self,
                                    vid_block,
                                    audio_block,
                                    vid,
                                    audio,
                                    vid_e,
                                    vid_seq_lens,
                                    vid_grid_sizes,
                                    vid_freqs,
                                    vid_context,
                                    vid_context_lens,
                                    audio_e,
                                    audio_seq_lens,
                                    audio_grid_sizes,
                                    audio_freqs,
                                    audio_context,
                                    audio_context_lens
                                    ):
        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], f"{audio_e.shape}, {audio.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        # audio self-attention
        audio_y = audio_block.self_attn(
            audio_block.norm1(audio).bfloat16() * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2), audio_seq_lens, audio_grid_sizes,
            audio_freqs)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e[2].squeeze(2)

        ## video modulation
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], f"{vid_e.shape}, {vid.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)

        # video self-attention
        vid_y = vid_block.self_attn(
            vid_block.norm1(vid).bfloat16() * (1 + vid_e[1].squeeze(2)) + vid_e[0].squeeze(2), vid_seq_lens, vid_grid_sizes,
            vid_freqs)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e[2].squeeze(2)

        og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e
        )

        return vid, audio

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False
    ):  

        assert clip_fea is None 
        assert y is None

        if vid is None or all([x is None for x in vid]):
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None)
        
        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean), None
        
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            vid, audio = gradient_checkpointing(
                    enabled=(self.training and self.gradient_checkpointing),
                    module=self.single_fusion_block_forward,
                    vid_block=vid_block,
                    audio_block=audio_block,
                    vid=vid,
                    audio=audio,
                    **kwargs
                )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e)

        return vid, audio

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()