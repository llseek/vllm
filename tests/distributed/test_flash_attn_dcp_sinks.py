# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention DCP must include one sink per query head."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig

from vllm.config import (
    AttentionConfig,
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
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.workspace import init_workspace_manager


def _run_dcp_sinks(rank: int, model_path: str, init_method: str, comm: str) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            dtype="bfloat16",
            skip_tokenizer_init=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=16,
            dcp_comm_backend=comm,
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=128,
            max_num_seqs=4,
            max_num_batched_tokens=128,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        attention_config=AttentionConfig(backend="FLASH_ATTN"),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(2, rank, init_method, rank)
        initialize_model_parallel(2, decode_context_model_parallel_size=2)
        init_workspace_manager(device)
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.bfloat16,
        )
        builder = FlashAttentionMetadataBuilder(spec, [], config, device)
        sinks = torch.arange(8, device=device, dtype=torch.bfloat16)[
            rank * 4 : (rank + 1) * 4
        ].contiguous()
        impl = FlashAttentionImpl(
            num_heads=4,
            head_size=64,
            scale=64**-0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            sinks=sinks,
        )
        # Empty context, an empty rank, unequal shards, and mixed decode/extend.
        for query_lens, context_lens in [
            ([1], [0]),
            ([4], [0]),
            ([1], [16]),
            ([4], [33]),
            ([1, 4], [1, 33]),
        ]:
            queries, keys, values, references = [], [], [], []
            key_cache = torch.zeros(8, 16, 1, 64, device=device, dtype=torch.bfloat16)
            value_cache = torch.zeros_like(key_cache)
            block_table = torch.zeros(
                len(query_lens), 4, device=device, dtype=torch.int32
            )
            for req, (query_len, context_len) in enumerate(
                zip(query_lens, context_lens)
            ):
                # Zero logits and unit values give the explicit sink formula,
                # without cancellation from intermediate BF16 attention states.
                q = torch.zeros(query_len, 4, 64, device=device, dtype=torch.bfloat16)
                k = torch.zeros(
                    context_len + query_len, 1, 64, device=device, dtype=q.dtype
                )
                v = torch.ones_like(k)
                count = context_len + torch.arange(1, query_len + 1, device=device)
                count = count[:, None, None].float()
                expected = count / (count + sinks.float().exp()[None, :, None])
                references.append(expected.expand_as(q))
                queries.append(q)
                keys.append(k[context_len:])
                values.append(v[context_len:])
                local = torch.arange(context_len, device=device)
                local = local[(local // 16) % 2 == rank]
                block_ids = torch.arange(req * 4, (req + 1) * 4, device=device)
                block_table[req] = block_ids
                key_cache[req * 4 : (req + 1) * 4].flatten(0, 1)[: len(local)] = k[
                    local
                ]
                value_cache[req * 4 : (req + 1) * 4].flatten(0, 1)[: len(local)] = v[
                    local
                ]
            query = torch.cat(queries)
            cu_cpu = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )
            seq_cpu = torch.tensor(context_lens, dtype=torch.int32) + torch.tensor(
                query_lens, dtype=torch.int32
            )
            common = CommonAttentionMetadata(
                query_start_loc=cu_cpu.to(device),
                query_start_loc_cpu=cu_cpu,
                seq_lens=seq_cpu.to(device),
                num_reqs=len(query_lens),
                num_actual_tokens=query.shape[0],
                max_query_len=max(query_lens),
                max_seq_len=int(seq_cpu.max()),
                block_table_tensor=block_table,
                slot_mapping=torch.zeros(
                    query.shape[0], device=device, dtype=torch.int64
                ),
                seq_lens_cpu_upper_bound=seq_cpu,
            )
            metadata = builder.build(0, common)
            output = torch.empty_like(query)
            impl._forward_with_dcp(
                query,
                torch.cat(keys),
                torch.cat(values),
                key_cache,
                value_cache,
                output,
                metadata,
            )
            torch.testing.assert_close(
                output.float(),
                torch.cat(references),
                rtol=2e-2,
                atol=2e-3,
            )
    cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("comm", ["ag_rs", "a2a"])
def test_flash_attn_dcp_sinks(tmp_path: Path, comm: str) -> None:
    if (
        not current_platform.is_cuda()
        or current_platform.get_device_capability()[0] < 9
    ):
        pytest.skip("Attention sinks require SM90 or newer")
    if torch.accelerator.device_count() < 2:
        pytest.skip("DCP requires two GPUs")
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    mp.spawn(
        _run_dcp_sinks,
        args=(str(tmp_path), f"file://{tmp_path / 'store'}", comm),
        nprocs=2,
    )


def _generate_sink_model(
    _rank: int, model_path: str, dcp_size: int, output_path: str
) -> None:
    import json

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
        attention_config={"backend": "FLASH_ATTN"},
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.04,
        max_logprobs=256,
        kv_cache_memory_bytes=32 * 1024**2,
        enable_prefix_caching=False,
    )
    prompts = [
        {"prompt_token_ids": [3 + i % 250 for i in range(n)]} for n in (1, 17, 33)
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=256),
        use_tqdm=False,
    )
    results = []
    for output in outputs:
        completion = output.outputs[0]
        results.append(
            {
                "tokens": list(completion.token_ids),
                "logprobs": [
                    [step[i].logprob for i in range(256)]
                    for step in completion.logprobs
                ],
            }
        )
    Path(output_path).write_text(json.dumps(results))


@pytest.mark.distributed(num_gpus=2)
def test_flash_attn_dcp_sinks_generation(tmp_path: Path) -> None:
    import json

    from transformers import GraniteConfig, GraniteForCausalLM

    if (
        not current_platform.is_cuda()
        or current_platform.get_device_capability()[0] < 9
    ):
        pytest.skip("Attention sinks require SM90 or newer")
    if torch.accelerator.device_count() < 2:
        pytest.skip("DCP requires two GPUs")
    torch.manual_seed(0)
    config = GraniteConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        head_dim=64,
        num_attention_heads=8,
        num_key_value_heads=1,
        vocab_size=256,
        max_position_embeddings=128,
        sliding_window=None,
    )
    config.update({"attention_sinks": True})
    model = GraniteForCausalLM(config).to(torch.bfloat16)
    state = model.state_dict()
    for layer in range(config.num_hidden_layers):
        state[f"model.layers.{layer}.self_attn.sinks"] = torch.full(
            (config.num_attention_heads,),
            4.0,
            dtype=torch.bfloat16,
        )
    model.save_pretrained(tmp_path / "model", state_dict=state)
    del model
    results = []
    for dcp_size in (1, 2):
        output_path = tmp_path / f"dcp{dcp_size}.json"
        mp.spawn(
            _generate_sink_model,
            args=(str(tmp_path / "model"), dcp_size, str(output_path)),
            nprocs=1,
        )
        results.append(json.loads(output_path.read_text()))
    for baseline, dcp in zip(*results):
        assert baseline["tokens"] == dcp["tokens"]
        torch.testing.assert_close(
            torch.tensor(dcp["logprobs"]),
            torch.tensor(baseline["logprobs"]),
            rtol=0,
            atol=1e-2,
        )
