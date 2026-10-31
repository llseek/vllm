# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 sink normalization across DCP KV shards."""

import math

import torch
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test


def _small_config(path):
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config

    HYV4Config(
        architectures=["HYV4ForCausalLM"],
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=1,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        index_n_heads=32,
        index_topk=2048,
        layer_types=["deepseek_sparse_attention"] * 2,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        vocab_size=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        learnable_sink=True,
    ).save_pretrained(path)


def _sink_worker(rank, model_path, init_method):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CUDAGraphMode,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        DCPGroupColumnParallelLinear,
    )
    from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseBackend
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs
    from vllm.v1.attention.ops.flashmla import get_mla_metadata
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            dtype="bfloat16",
            max_model_len=128,
            skip_tokenizer_init=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla", block_size=64),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=128,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(2, rank, init_method, rank)
        initialize_model_parallel(2, decode_context_model_parallel_size=2)
        init_workspace_manager(device)
        sinks = torch.linspace(-3, 5, 32, device=device).chunk(2)[rank].contiguous()
        with torch.device(device):
            kv_proj = ColumnParallelLinear(
                512,
                32 * (192 + 256),
                bias=False,
                params_dtype=torch.bfloat16,
            )
            q_proj = DCPGroupColumnParallelLinear(
                8,
                32 * 576,
                bias=False,
                params_dtype=torch.bfloat16,
            )
        torch.nn.init.zeros_(q_proj.weight)
        indices_buffer = torch.full((1, 128), -1, device=device, dtype=torch.int32)
        with torch.device(device):
            layer = MLAAttention(
                num_heads=16,
                scale=256**-0.5,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
                q_lora_rank=128,
                kv_lora_rank=512,
                kv_b_proj=kv_proj,
                cache_config=config.cache_config,
                prefix="layers.0.attn",
                attn_backend=HYV4FlashMLASparseBackend,
                use_sparse=True,
                topk_indices_buffer=indices_buffer,
                sinks=sinks,
            )
        impl = layer.impl
        impl.process_weights_after_loading(torch.bfloat16)
        q_local = torch.zeros(1, 16, 576, device=device, dtype=torch.bfloat16)
        gathered = get_dcp_group().all_gather(q_local, dim=1)
        with torch.inference_mode():
            replicated, _ = q_proj(
                torch.ones(1, 8, device=device, dtype=torch.bfloat16)
            )
        replicated = replicated.reshape_as(gathered)
        torch.testing.assert_close(replicated, gathered, rtol=0, atol=0)

        cache = torch.zeros(1, 64, 656, device=device, dtype=torch.uint8)
        # Uniform unit values and zero keys give a closed-form attention result.
        cache[..., :512] = torch.ones(
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ).view(torch.uint8)
        cache[..., 512:528] = torch.ones(
            4,
            device=device,
            dtype=torch.float32,
        ).view(torch.uint8)
        scheduler, _ = get_mla_metadata()
        metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler,
            dummy_block_table=torch.zeros(1, 1, device=device, dtype=torch.int32),
            cache_lens=torch.full((1,), 64, device=device, dtype=torch.int32),
        )
        attn_metadata = FlashMLASparseMetadata(
            num_reqs=1,
            max_query_len=1,
            max_seq_len=64,
            num_actual_tokens=1,
            query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
            slot_mapping=torch.zeros(1, device=device, dtype=torch.int64),
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            block_table=torch.zeros(1, 1, device=device, dtype=torch.int32),
            req_id_per_token=torch.zeros(1, device=device, dtype=torch.int32),
            seq_lens=torch.full((1,), 64, device=device, dtype=torch.int32),
            fp8_extra_metadata=metadata,
            fp8_use_mixed_batch=True,
            cp_kv_cache_interleave_size=1,
        )
        for q in (gathered, replicated):
            for counts in ((8, 24), (0, 32)):
                selected = torch.cat(
                    [
                        2 * torch.arange(n, device=device, dtype=torch.int32) + r
                        for r, n in enumerate(counts)
                    ]
                )
                indices_buffer.fill_(-1)
                indices_buffer[:, : len(selected)] = selected
                output, lse = impl.forward_mqa(q, cache, attn_metadata, layer)
                assert lse is not None
                merged = cp_lse_ag_out_rs(
                    output.contiguous(),
                    lse.contiguous(),
                    get_dcp_group(),
                )
                expected = sum(counts) / (sum(counts) + sinks.exp())
                torch.testing.assert_close(
                    merged[0, :, 0].float(),
                    expected,
                    rtol=1e-2,
                    atol=2e-3,
                    msg=f"rank={rank}, counts={counts}",
                )
    cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
def test_sink_dcp_merge(tmp_path):
    """Count the sink once, including unequal and empty local KV shards."""
    _small_config(tmp_path)
    mp.spawn(_sink_worker, args=(str(tmp_path), f"file://{tmp_path}/store"), nprocs=2)


@multi_gpu_test(num_gpus=2)
def test_sink_dcp_generation(vllm_runner, tmp_path):
    """DCP generation agrees with the same TP layout without KV sharding."""
    _small_config(tmp_path)
    results = []
    for dcp in (1, 2):
        with vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            load_format="dummy",
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp,
            dtype="bfloat16",
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            attention_backend="FLASHMLA_SPARSE",
            enforce_eager=True,
            enable_prefix_caching=False,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            gpu_memory_utilization=0.2,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            seed=17,
        ) as model:
            from vllm import SamplingParams

            outputs = model.llm.generate(
                [
                    {"prompt_token_ids": [11, 12, 13, 14]},
                    {"prompt_token_ids": list(range(20, 84))},
                ],
                SamplingParams(
                    temperature=0, max_tokens=4, ignore_eos=True, logprobs=5
                ),
            )
            result = []
            for output in outputs:
                completion = output.outputs[0]
                assert len(completion.token_ids) == 4
                assert math.isfinite(completion.cumulative_logprob)
                result.append(
                    (
                        list(completion.token_ids),
                        [
                            step[token].logprob
                            for token, step in zip(
                                completion.token_ids, completion.logprobs
                            )
                        ],
                    )
                )
            results.append(result)
    for reference, actual in zip(*results):
        assert actual[0] == reference[0]
        torch.testing.assert_close(
            torch.tensor(actual[1]),
            torch.tensor(reference[1]),
            rtol=0,
            atol=0.03,
        )
