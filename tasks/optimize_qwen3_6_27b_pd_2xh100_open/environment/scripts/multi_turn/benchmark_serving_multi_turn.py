#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections import Counter, deque
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from statistics import mean
from typing import NamedTuple

import aiohttp  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from bench_dataset import (
    ConversationsMap,
    ConvId,
    GenConvArgs,
    MessagesList,
    ShareGptConversations,
    conversations_dict_to_list,
    conversations_list_to_dict,
    generate_conversations,
    parse_input_json_file,
)
from bench_utils import TEXT_SEPARATOR, Color, logger
from transformers import AutoTokenizer  # type: ignore

NUM_TOKENS_FROM_DATASET = 0
TERM_SIGNAL = None


class ConversationSampling(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"

    def __str__(self):
        return self.value


class ClientArgs(NamedTuple):
    seed: int
    max_num_requests: int | None
    skip_first_turn: bool
    max_turns: int | None
    max_active_conversations: int
    verbose: bool
    print_content: bool
    verify_output: bool
    conversation_sampling: ConversationSampling
    request_rate: float
    max_retries: int


class RequestArgs(NamedTuple):
    chat_url: str
    model: str
    stream: bool
    limit_min_tokens: int
    limit_max_tokens: int
    timeout_sec: int


class BenchmarkArgs(NamedTuple):
    url: str
    num_clients: int
    early_stop: bool


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    first_chunk: str
    content: str
    num_chunks: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"  # noqa: E501


class RequestStats(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    conversation_id: str
    client_id: int

    def __str__(self) -> str:
        return (
            f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}, input_num_tokens {self.input_num_tokens}, "  # noqa: E501
            f"output_num_tokens {self.output_num_tokens} ({self.output_num_chunks} chunks, {self.output_num_first_chunk_tokens} tokens in first chunk), "  # noqa: E501
            f"approx_cached_percent {self.approx_cached_percent:.2f}%"
        )


class MetricStats:
    def __init__(self) -> None:
        self.min: float | None = None
        self.max: float | None = None
        self.avg: float | None = None
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        if self.min is None:
            self.min = value
        else:
            self.min = min(self.min, value)

        if self.max is None:
            self.max = value
        else:
            self.max = max(self.max, value)

        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f}, min: {self.min:>10.3f}, max: {self.max:>10.3f}"


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.index = 0
        self.sum = 0.0
        self.count = 0
        self.avg: float | None = None

    def update(self, new_value: float) -> None:
        if self.count < self.window_size:
            self.sum += new_value
            self.window[self.count] = new_value
            self.count += 1
        else:
            old_value = self.window[self.index]
            self.sum = self.sum - old_value + new_value
            self.window[self.index] = new_value
            self.index = (self.index + 1) % self.window_size

        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f} ({self.count} samples)"


class DebugStats:
    def __init__(self, logger: logging.Logger, window_size: int) -> None:
        self.logger = logger
        self.metrics: dict[str, MovingAverage | MetricStats] = {
            "moving_avg_ttft_ms": MovingAverage(window_size),
            "moving_avg_tpot_ms": MovingAverage(window_size),
            "ttft_ms": MetricStats(),
            "tpot_ms": MetricStats(),
            "latency_ms": MetricStats(),
            "input_num_turns": MetricStats(),
            "input_num_tokens": MetricStats(),
            "output_num_tokens": MetricStats(),
        }

    def update(self, data: RequestStats) -> None:
        self.metrics["ttft_ms"].update(data.ttft_ms)
        self.metrics["moving_avg_ttft_ms"].update(data.ttft_ms)
        self.metrics["tpot_ms"].update(data.tpot_ms)
        self.metrics["moving_avg_tpot_ms"].update(data.tpot_ms)
        self.metrics["latency_ms"].update(data.latency_ms)
        self.metrics["input_num_turns"].update(data.input_num_turns)
        self.metrics["input_num_tokens"].update(data.input_num_tokens)
        self.metrics["output_num_tokens"].update(data.output_num_tokens)

    def print(self) -> None:
        self.logger.info("-" * 50)
        for k, v in self.metrics.items():
            kv_info = f"[{k:25}] {v}"
            self.logger.info(kv_info)
        self.logger.info("-" * 50)


def nanosec_to_millisec(value: float) -> float:
    return value / 1000000.0


def nanosec_to_sec(value: float) -> float:
    return value / 1000000000.0


async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    stream: bool = True,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
) -> ServerResponse:
    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": 0.0,
    }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": False}

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}

    if max_tokens is not None:
        token_based_timeout = int(max_tokens * 0.2)
        if token_based_timeout > timeout_sec:
            timeout_sec = token_based_timeout
            logger.info(
                "Using timeout of %ds based on max_tokens %d",
                timeout_sec,
                max_tokens,
            )
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft: float | None = None
    chunk_delay: list[int] = []
    latency: float | None = None
    first_chunk = ""
    generated_text = ""

    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(
        url=chat_url, json=payload, headers=headers, timeout=timeout
    ) as response:
        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += message["content"]
                else:
                    timestamp: int = time.perf_counter_ns()
                    data = json.loads(chunk)

                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if ttft is None:
                            first_token_time = time.perf_counter_ns()
                            ttft = first_token_time - start_time
                            first_chunk = delta["content"]
                        else:
                            chunk_delay.append(timestamp - most_recent_timestamp)

                        generated_text += delta["content"]

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            logger.warning(
                f"{Color.YELLOW}Received HTTP status {http_status.value} "
                f"({http_status.phrase}): {content}{Color.RESET}"
            )

    if latency is None:
        latency = -1.0
        if valid_response:
            latency = time.perf_counter_ns() - start_time

    if ttft is None:
        ttft = latency

    tpot: float = mean(chunk_delay) if len(chunk_delay) > 0 else 0.0
    num_chunks: int = len(chunk_delay)

    sr = ServerResponse(
        valid=valid_response,
        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
        tpot_ms=nanosec_to_millisec(tpot),
        latency_ms=nanosec_to_millisec(latency),
        start_time_ms=nanosec_to_millisec(start_time),
        first_chunk=first_chunk,
        content=generated_text,
        num_chunks=num_chunks,
    )
    return sr


def get_short_string(input: str) -> str:
    n = 20
    if len(input) < 400:
        return input

    return f"{input[:n]}...{input[-n:]}"


def get_token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_messages_token_count(
    tokenizer: AutoTokenizer, messages: list[dict[str, str]]
) -> int:
    token_count = 0
    for m in messages:
        token_count += get_token_count(tokenizer, m["content"])

    return token_count


async def send_turn(
    session: aiohttp.ClientSession,
    client_id: int,
    conv_id: str,
    conversation_messages: MessagesList,
    messages_to_use: int,
    tokenizer: AutoTokenizer,
    req_args: RequestArgs,
    verbose: bool,
    verify_output: bool,
) -> RequestStats | None:
    assert messages_to_use > 0
    assert messages_to_use <= len(conversation_messages)

    messages = conversation_messages[:messages_to_use]

    index = messages_to_use - 1

    assert len(messages[index].keys()) == 2
    assert "role" in messages[index] and "content" in messages[index]
    assert messages[index]["role"] == "user", (
        f"Failed on conversation ID {conv_id}, message role should be user"
    )

    if verbose:
        print(
            f"{Color.CYAN}Messages (conversation ID {conv_id},"
            f" {len(messages)} turns):{Color.RESET}",
            messages,
        )

    min_tokens = None if req_args.limit_min_tokens < 0 else req_args.limit_min_tokens
    max_tokens = None if req_args.limit_max_tokens < 0 else req_args.limit_max_tokens

    if len(conversation_messages) > messages_to_use:
        if (
            min_tokens == NUM_TOKENS_FROM_DATASET
            or max_tokens == NUM_TOKENS_FROM_DATASET
        ):
            assistant_answer = conversation_messages[messages_to_use]
            answer_num_tokens = get_token_count(tokenizer, assistant_answer["content"])
            assert assistant_answer["role"] == "assistant"

        if min_tokens == NUM_TOKENS_FROM_DATASET:
            min_tokens = max(1, answer_num_tokens)

        if max_tokens == NUM_TOKENS_FROM_DATASET:
            max_tokens = max(1, answer_num_tokens)

    response: ServerResponse = await send_request(
        session,
        messages,
        req_args.chat_url,
        req_args.model,
        req_args.stream,
        min_tokens,
        max_tokens,
        req_args.timeout_sec,
    )

    if response.valid is False:
        return None

    input_num_tokens = get_messages_token_count(tokenizer, messages)

    question_num_tokens = get_token_count(tokenizer, messages[index]["content"])

    assert input_num_tokens >= question_num_tokens
    history_num_tokens = input_num_tokens - question_num_tokens

    first_chunk_tokens = get_token_count(tokenizer, response.first_chunk)

    output_content = response.content
    output_num_tokens = get_token_count(tokenizer, output_content)

    approx_cached_percent = (
        100.0 * (history_num_tokens / input_num_tokens) if input_num_tokens > 0 else 0.0
    )

    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        decode_ms = response.latency_ms - response.ttft_ms
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        tpot_ms = decode_ms / decode_num_tokens
    else:
        tpot_ms = 0.0

    if first_chunk_tokens > 1:
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response.ttft_ms - delta_ms)
    else:
        ttft_ms = response.ttft_ms

    rs = RequestStats(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=response.latency_ms,
        start_time_ms=response.start_time_ms,
        input_num_turns=len(messages),
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        output_num_chunks=response.num_chunks,
        output_num_first_chunk_tokens=first_chunk_tokens,
        approx_cached_percent=approx_cached_percent,
        conversation_id=conv_id,
        client_id=client_id,
    )

    if verbose:
        print(
            f"\n{Color.YELLOW}Response ({output_num_tokens} tokens):{Color.RESET}",
            output_content,
        )
        print(f"{Color.YELLOW}Response metrics: {rs}{Color.RESET}")
        print("-" * 70)

    answer_index = messages_to_use
    if len(conversation_messages) > answer_index:
        assert conversation_messages[answer_index]["role"] == "assistant", (
            f"Failed on conversation ID {conv_id}, message role should be assistant"
        )

        orig_content = conversation_messages[answer_index]["content"]
        if verify_output:
            debug_info = (
                f"LLM/dataset answers do not match ({conv_id}):"
                f"\n'{get_short_string(output_content)}' (len: {len(output_content)}),"
                f"\n'{get_short_string(orig_content)}' (len: {len(orig_content)})"
            )
            if orig_content != output_content:
                raise ValueError(debug_info)

        conversation_messages[answer_index]["content"] = output_content
    else:
        new_answer = {"role": "assistant", "content": output_content}
        conversation_messages.append(new_answer)

    return rs


async def poisson_sleep(request_rate: float, verbose: bool = False) -> None:
    assert request_rate > 0

    interval = np.random.exponential(1.0 / request_rate)
    if verbose:
        logger.info(f"Sleeping for {interval:.3f} seconds...")
    await asyncio.sleep(interval)


async def exponential_backoff_sleep(
    attempt_cnt: int,
    base_rate: float = 1.0,
    backoff_factor: float = 2.0,
    jitter_fraction: float = 0.10,
    verbose: bool = False,
) -> None:
    backoff_delay = base_rate * (backoff_factor**attempt_cnt)
    jittered_delay = backoff_delay * (
        1 + np.random.uniform(-jitter_fraction, jitter_fraction)
    )

    if verbose:
        logger.info(f"Backoff for {jittered_delay:.3f} seconds...")

    await asyncio.sleep(jittered_delay)


async def client_main(
    args: ClientArgs,
    req_args: RequestArgs,
    client_id: int,
    tokenizer: AutoTokenizer,
    stop_event: mp.Event,  # type: ignore
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    logger.info(
        f"{Color.CYAN}Started client {client_id}: max_num_requests={args.max_num_requests}, max_active_conversations={args.max_active_conversations}{Color.RESET}"  # noqa: E501
    )

    client_seed = args.seed + client_id + 1
    random.seed(client_seed)
    np.random.seed(client_seed)

    active_convs: ConversationsMap = {}
    conv_id_queue: deque = deque(maxlen=args.max_active_conversations)

    turns_count: Counter = Counter()
    num_successes = 0
    num_failures = 0

    time_of_last_turn: dict[ConvId, float] = {}

    task_queue_empty = False

    async with aiohttp.ClientSession() as session:

        while task_queue_empty is False:
            result = None

            if (
                args.max_num_requests
                and num_successes + num_failures == args.max_num_requests
            ):
                logger.info(
                    f"{Color.YELLOW}Client {client_id} reached "
                    f"request limit{Color.RESET}"
                )
                break

            if stop_event.is_set():  # type: ignore
                logger.info(
                    f"{Color.YELLOW}Client {client_id} received "
                    f"a termination signal{Color.RESET}"
                )
                break

            while (
                len(active_convs) < args.max_active_conversations
                and task_queue_empty is False
            ):
                conv_id, messages = task_queue.get()

                if conv_id is TERM_SIGNAL:
                    task_queue_empty = True
                    break

                if args.skip_first_turn:
                    turns_count[conv_id] += 2

                if turns_count[conv_id] < len(messages):
                    active_convs[conv_id] = messages
                    conv_id_queue.append(conv_id)

                    if args.verbose:
                        logger.info(
                            f"{Color.GREEN}Client {client_id} will use conversation ID {conv_id} (active conversations {len(active_convs)}){Color.RESET}"  # noqa: E501
                        )

                elif args.verbose:
                    logger.info(
                        f"{Color.YELLOW}Client {client_id} will not use conversation ID {conv_id} (all {len(messages)} messages already sent){Color.RESET}"  # noqa: E501
                    )

            if len(active_convs) == 0 or task_queue_empty:
                logger.info(
                    f"{Color.YELLOW}Client {client_id} has no more work{Color.RESET}"
                )
                break

            if args.conversation_sampling == ConversationSampling.ROUND_ROBIN:
                conv_id = conv_id_queue.pop()
            else:
                active_ids = list(active_convs.keys())
                conv_id = random.choice(active_ids)

            messages = active_convs[conv_id]
            assert isinstance(messages, list) and len(messages) > 0

            turns_count[conv_id] += 1
            current_turn = turns_count[conv_id]

            assert current_turn < len(messages), (
                f"Turn number {current_turn} is invalid for conversation ID {conv_id}"
                f" that has only {len(messages)} messages"
            )

            if args.verbose:
                curr_time_sec: float = time.perf_counter()
                time_since_last_turn: str | float = "N/A"
                if conv_id in time_of_last_turn:
                    time_since_last_turn = round(
                        curr_time_sec - time_of_last_turn[conv_id], 3
                    )
                logger.info(
                    f"Client {client_id} using conversation ID {conv_id} (turn: {current_turn}, time since last turn [sec]: {time_since_last_turn})"  # noqa: E501
                )
                time_of_last_turn[conv_id] = curr_time_sec

            success = False
            for attempt_cnt in range(args.max_retries + 1):
                try:
                    exception = False
                    result = await send_turn(
                        session,
                        client_id,
                        conv_id,
                        messages,
                        current_turn,
                        tokenizer,
                        req_args,
                        args.print_content,
                        args.verify_output,
                    )
                    if result is not None:
                        result_queue.put(result)
                        success = True
                        break
                    else:
                        logger.warning(
                            f"{Color.YELLOW}Client {client_id} - Request rejected during conversation ID {conv_id} (turn: {current_turn}){Color.RESET}"  # noqa: E501
                        )
                except asyncio.exceptions.TimeoutError:
                    exception = True
                    logger.error(
                        "%sClient %d - Timeout during conversation ID %s (turn: %d). "
                        "Base timeout is %ss (set with --request-timeout-sec), but the "
                        "effective timeout may be longer based on max_tokens. If this "
                        "is unexpected, consider increasing the timeout or checking "
                        "model performance.%s",
                        Color.RED,
                        client_id,
                        conv_id,
                        current_turn,
                        req_args.timeout_sec,
                        Color.RESET,
                    )
                except Exception:
                    exception = True
                    logger.exception(
                        f"{Color.RED}Client {client_id} - Exception during conversation ID {conv_id} (turn: {current_turn}){Color.RESET}"  # noqa: E501
                    )

                if not success and attempt_cnt < args.max_retries:
                    await exponential_backoff_sleep(attempt_cnt, verbose=args.verbose)

            if not success:
                num_failures += 1
                active_convs.pop(conv_id)
                if exception:
                    break

            else:
                num_successes += 1

                turns_count[conv_id] += 1

                max_turns = len(messages)
                if args.max_turns is not None:
                    max_turns = min(args.max_turns, max_turns)

                if turns_count[conv_id] >= max_turns:
                    conv_queue.put((conv_id, active_convs.pop(conv_id)))
                    if args.verbose:
                        logger.info(
                            f"{Color.GREEN}Client {client_id} finished "
                            f"conversation ID {conv_id}{Color.RESET}"
                        )
                else:
                    conv_id_queue.appendleft(conv_id)

            if args.request_rate > 0:
                await poisson_sleep(args.request_rate, args.verbose)

    conv_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    logger.info(
        f"{Color.CYAN}Client {client_id} is done "
        f"({num_successes=}, {num_failures=}){Color.RESET}"
    )


def worker_function(
    client_id: int,
    tokenizer: AutoTokenizer,
    client_args: ClientArgs,
    req_args: RequestArgs,
    stop_event: mp.Event,  # type: ignore
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    conv_queue: mp.Queue,
) -> None:
    asyncio.run(
        client_main(
            client_args,
            req_args,
            client_id,
            tokenizer,
            stop_event,
            task_queue,
            result_queue,
            conv_queue,
        )
    )


def get_client_config(
    args: argparse.Namespace, input_conv: ConversationsMap
) -> tuple[ClientArgs, RequestArgs]:
    if args.num_clients < 1:
        raise ValueError("Number of clients must be a positive number")

    if len(input_conv) < args.num_clients:
        raise ValueError(
            "Number of conversations must be equal or larger than the number of clients"
        )

    max_req_per_client: int | None = None
    if args.max_num_requests is not None:
        req_per_client = args.max_num_requests // args.num_clients
        if req_per_client < 1:
            raise ValueError("Number of requests should be at least one per client")
        max_req_per_client = req_per_client

    max_active_conversations = args.max_active_conversations
    if max_active_conversations is None:
        max_active_conversations = args.num_clients

    if max_active_conversations > len(input_conv):
        raise ValueError(
            f"Max active conversations {max_active_conversations} "
            "must be equal or less than the total number of conversations"
        )

    max_active_conv_per_client = max_active_conversations // args.num_clients
    if max_active_conv_per_client < 1:
        raise ValueError(
            f"Max active conversations {max_active_conversations} "
            "must be equal or greater than the number of clients"
        )

    skip_first_turn = args.warmup_step

    client_args = ClientArgs(
        seed=args.seed,
        max_num_requests=max_req_per_client,
        skip_first_turn=skip_first_turn,
        max_turns=args.max_turns,
        max_active_conversations=max_active_conv_per_client,
        verbose=args.verbose,
        print_content=args.print_content,
        verify_output=args.verify_output,
        conversation_sampling=args.conversation_sampling,
        request_rate=args.request_rate,
        max_retries=args.max_retries,
    )

    if args.limit_min_tokens > 0 or args.limit_max_tokens > 0:
        if args.limit_min_tokens < 1 or args.limit_max_tokens < 1:
            raise ValueError(
                "Invalid min/max tokens limits (both limits should be provided)"
            )
        if args.limit_min_tokens > args.limit_max_tokens:
            raise ValueError(
                "Invalid min/max tokens limits (min should not be larger than max)"
            )

    if args.request_timeout_sec <= 0:
        raise ValueError("Request timeout must be a positive number")

    chat_url = f"{args.url}/v1/chat/completions"
    model_name = args.served_model_name if args.served_model_name else args.model

    req_args = RequestArgs(
        chat_url=chat_url,
        model=model_name,
        stream=not args.no_stream,
        limit_min_tokens=args.limit_min_tokens,
        limit_max_tokens=args.limit_max_tokens,
        timeout_sec=args.request_timeout_sec,
    )

    return client_args, req_args


async def main_mp(
    client_args: ClientArgs,
    req_args: RequestArgs,
    bench_args: BenchmarkArgs,
    tokenizer: AutoTokenizer,
    input_conv: ConversationsMap,
) -> tuple[ConversationsMap, list[RequestStats]]:
    stop_event = mp.Event()

    task_queue: mp.Queue = mp.Queue()

    result_queue: mp.Queue = mp.Queue()

    conv_queue: mp.Queue = mp.Queue()
    output_conv: ConversationsMap = {}
    client_metrics: list[RequestStats] = []

    start_time = time.perf_counter_ns()
    logger.info(f"{Color.GREEN}Starting {bench_args.num_clients} clients{Color.RESET}")

    clients = []
    for client_id in range(bench_args.num_clients):
        client = mp.Process(
            name=f"client_{client_id}",
            target=worker_function,
            args=(
                client_id,
                tokenizer,
                client_args,
                req_args,
                stop_event,
                task_queue,
                result_queue,
                conv_queue,
            ),
        )
        clients.append(client)
        client.start()

    for conv_id, messages in input_conv.items():
        task_queue.put((conv_id, messages))

    for _ in range(bench_args.num_clients):
        task_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    num_clients_finished = 0
    total_convs = len(input_conv)

    debug_stats = DebugStats(logger, min(15 * bench_args.num_clients, 500))

    while num_clients_finished < bench_args.num_clients:
        conv_id, messages = conv_queue.get()

        while not result_queue.empty():
            new_data = result_queue.get()
            client_metrics.append(new_data)
            debug_stats.update(new_data)

        if conv_id is TERM_SIGNAL:
            num_clients_finished += 1
            logger.info(
                f"{Color.CYAN}{num_clients_finished} out of "
                f"{bench_args.num_clients} clients finished{Color.RESET}"
            )

            if bench_args.early_stop and not stop_event.is_set():
                logger.info(
                    f"{Color.YELLOW}Sending termination signal to clients{Color.RESET}"
                )
                stop_event.set()
        else:
            output_conv[conv_id] = messages

            finished_convs = len(output_conv)
            percent = finished_convs / total_convs

            print_cycle = max(3, int(bench_args.num_clients / 4))

            if finished_convs % print_cycle == 0:
                runtime_sec = nanosec_to_sec(time.perf_counter_ns() - start_time)
                logger.info(
                    f"{Color.CYAN}Finished {finished_convs} out of {total_convs} conversations ({percent:.0%}), "  # noqa: E501
                    f"{num_clients_finished} out of {bench_args.num_clients} clients finished, collected {len(client_metrics)} measurements, runtime {runtime_sec:.3f} sec{Color.RESET}"  # noqa: E501
                )

                rps: str | float = round(len(client_metrics) / runtime_sec, 3)
                if len(client_metrics) < (5 * bench_args.num_clients):
                    rps = "N/A"

                runtime_left_sec: str | float = round(
                    (runtime_sec / finished_convs) * (total_convs - finished_convs), 3
                )
                if percent < 0.05:
                    runtime_left_sec = "N/A"

                logger.info(
                    f"{Color.CYAN}Estimated req/sec {rps}, estimated runtime left {runtime_left_sec} sec{Color.RESET}"  # noqa: E501
                )
                debug_stats.print()

    logger.info(
        f"{Color.CYAN}All {bench_args.num_clients} clients finished{Color.RESET}"
    )

    while not result_queue.empty():
        client_metrics.append(result_queue.get())

    logger.info(f"Collected {len(client_metrics)} samples from all the clients")

    for client in clients:
        logger.info(
            f"{Color.CYAN}Waiting for client {client.name} "
            f"(is alive: {client.is_alive()}){Color.RESET}"
        )

        client.join(timeout=req_args.timeout_sec + 1)

        if client.is_alive():
            logger.warning(
                f"{Color.YELLOW}Client {client.name} will be terminated{Color.RESET}"
            )
            client.terminate()

        exitcode = client.exitcode
        if exitcode != 0:
            logger.error(
                f"{Color.RED}Client {client.name} exited "
                f"with exit code {exitcode}{Color.RESET}"
            )

    logger.info(
        f"All {bench_args.num_clients} clients exited (successfully "
        f"finished {len(output_conv)} out of {total_convs} conversations)"
    )

    unfinished_tasks = 0
    while not task_queue.empty():
        task_queue.get()
        unfinished_tasks += 1

    if unfinished_tasks > 0:
        logger.debug(f"Discarding {unfinished_tasks} unfinished tasks")

    task_queue.close()
    task_queue.join_thread()

    result_queue.close()
    result_queue.join_thread()

    conv_queue.close()
    conv_queue.join_thread()

    return output_conv, client_metrics


def get_filename_with_timestamp(label: str, extension: str) -> str:
    time_now = datetime.now()
    timestamp = time_now.strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"{label}__{timestamp}.{extension}"
    return filename


def process_statistics(
    client_metrics: list[RequestStats],
    warmup_percentages: list[float],
    test_params: dict,
    verbose: bool,
    gen_conv_args: GenConvArgs | None = None,
    excel_output: bool = False,
    warmup_runtime_sec: float | None = None,
) -> None:
    if len(client_metrics) == 0:
        logger.info("No samples to process")
        return

    logger.info(f"Processing {len(client_metrics)} samples...")

    raw_data = pd.DataFrame(client_metrics)

    if verbose:
        raw_data = raw_data.sort_values(by=["conversation_id", "start_time_ms"])
        raw_data["time_between_user_turns_sec"] = raw_data.groupby("conversation_id")[
            "start_time_ms"
        ].diff()

        raw_data["time_between_user_turns_sec"] = (
            raw_data["time_between_user_turns_sec"] / 1000.0
        )

    raw_data = raw_data.sort_values(by=["start_time_ms"])
    raw_data["end_time_ms"] = raw_data["start_time_ms"] + raw_data["latency_ms"]

    percentiles = [0.25, 0.5, 0.75, 0.9]

    if len(raw_data) >= 100:
        percentiles.append(0.99)

    if len(raw_data) >= 1000:
        percentiles.append(0.999)

    if len(raw_data) >= 10000:
        percentiles.append(0.9999)

    pd.set_option("display.precision", 2)

    exclude = [
        "start_time_ms",
        "end_time_ms",
        "output_num_first_chunk_tokens",
        "approx_cached_percent",
        "conversation_id",
        "client_id",
    ]

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Parameters:{Color.RESET}")
    for k, v in test_params.items():
        print(f"{k}={v}")

    if gen_conv_args is not None:
        gen_params = {
            "text_files": ", ".join(gen_conv_args.text_files),
            "input_num_turns": str(gen_conv_args.input_num_turns),
            "input_common_prefix_num_tokens": str(
                gen_conv_args.input_common_prefix_num_tokens
            ),
            "input_prefix_num_tokens": str(gen_conv_args.input_prefix_num_tokens),
            "input_num_tokens": str(gen_conv_args.input_num_tokens),
            "output_num_tokens": str(gen_conv_args.output_num_tokens),
        }

        print(f"{Color.YELLOW}Conversations Generation Parameters:{Color.RESET}")
        for k, v in gen_params.items():
            print(f"{k}={v}")

    print(TEXT_SEPARATOR)

    params_list = []
    df_list = []
    for percent in warmup_percentages:
        warmup_count = int(percent * len(raw_data))
        tail_count = len(raw_data) - warmup_count
        if tail_count == 0:
            break

        df = raw_data.tail(tail_count)

        runtime_sec = df["end_time_ms"].iloc[-1] - df["start_time_ms"].iloc[0]

        runtime_sec = runtime_sec / 1000.0
        requests_per_sec = float(len(df)) / runtime_sec
        params = {
            "runtime_sec": runtime_sec,
            "requests_per_sec": requests_per_sec,
        }
        if warmup_runtime_sec is not None:
            params["warmup_runtime_sec"] = warmup_runtime_sec
            params["total_runtime_incl_warmup_sec"] = runtime_sec + warmup_runtime_sec

        df = df.drop(columns=exclude).describe(percentiles=percentiles).transpose()

        params_list.append(params)
        df_list.append(df)

        if percent > 0 or len(warmup_percentages) > 1:
            print(
                f"{Color.YELLOW}Statistics summary "
                f"(assuming {percent:.0%} warmup samples):{Color.RESET}"
            )
        else:
            print(f"{Color.YELLOW}Statistics summary:{Color.RESET}")

        for k, v in params.items():
            if isinstance(v, float):
                print(f"{k} = {v:.3f}")
            else:
                print(f"{k} = {v}")
        print(TEXT_SEPARATOR)
        print(df)
        print(TEXT_SEPARATOR)

    if excel_output:
        prefix = f"statistics_{test_params['num_clients']}_clients"
        filename = get_filename_with_timestamp(prefix, "xlsx")

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            startrow = 0
            test_params_df = pd.DataFrame([test_params])
            test_params_df.to_excel(
                writer, sheet_name="Summary", index=False, startrow=startrow
            )
            startrow += len(test_params_df) + 3

            if gen_conv_args is not None:
                gen_params_df = pd.DataFrame([gen_params])
                gen_params_df.to_excel(
                    writer, sheet_name="Summary", index=False, startrow=(startrow - 1)
                )
                startrow += len(gen_params_df) + 3

            for params, df_stats in zip(params_list, df_list):
                df_params = pd.DataFrame([params])
                df_params.to_excel(
                    writer, sheet_name="Summary", index=False, startrow=startrow
                )
                startrow += len(df_params) + 2
                df_stats.to_excel(
                    writer, sheet_name="Summary", index=True, startrow=startrow
                )
                startrow += len(df_stats) + 3

            raw_data.to_excel(writer, sheet_name="Raw data", index=False, startrow=0)

        logger.info(
            f"{Color.GREEN}Client metrics exported to file: {filename}{Color.RESET}"
        )


async def get_server_info(url: str) -> None:
    logger.info(f"{Color.BLUE}Collecting information from server: {url}{Color.RESET}")
    async with aiohttp.ClientSession() as session:
        url_version = f"{url}/version"
        async with session.get(url_version) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Server version: {text}{Color.RESET}")

        url_models = f"{url}/v1/models"
        async with session.get(url_models) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Models:{Color.RESET}")
                models_data = json.loads(text)
                models_list = models_data["data"]
                for model in models_list:
                    model_id = model["id"]
                    max_model_len = model.get("max_model_len", "N/A")
                    logger.info(
                        f"{Color.BLUE}\t{model_id=}, {max_model_len=}{Color.RESET}"
                    )
            else:
                logger.info(f"{Color.RED}Failed to get models{Color.RESET}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Benchmark serving with multi-turn conversations",
        description="Benchmark online inference using REST API",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with ShareGPT conversations or "
        "configuration file for generation of synthetic conversations",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file containing conversations with updated assistant answers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generators (default: 0)",
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path of the LLM model"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the `--model` argument. ",
    )

    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the LLM API server",
    )

    parser.add_argument(
        "-p",
        "--num-clients",
        type=int,
        default=1,
        help="Number of clients that will send requests in parallel",
    )
    parser.add_argument(
        "-k",
        "--max-active-conversations",
        type=int,
        default=None,
        help="Max number of active conversations at a time (for all clients)",
    )
    parser.add_argument(
        "-n",
        "--max-num-requests",
        type=int,
        default=None,
        help="Max number of requests to send (total for all clients)",
    )

    parser.add_argument(
        "--warmup-step",
        default=False,
        action="store_true",
        help="Run a warmup step (using only the first turn of every conversation), "
        "measurements will not be included in the final benchmark results",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns/messages per conversation, "
        "includes both user and assistant messages "
        "(a positive number, e.g: 2, 4, 6, etc.), disabled by default",
    )
    parser.add_argument(
        "--no-early-stop",
        default=False,
        action="store_true",
        help="By default, the benchmark will stop if at least one client exits."
        " Use this flag to disable this behavior",
    )

    parser.add_argument(
        "--limit-max-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set max_tokens for the output token count of each request "
        "(must also set --limit-min-tokens). "
        "Overrides output token count from the input dataset. "
        "Use a negative value to disable this limit.",
    )
    parser.add_argument(
        "--limit-min-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set min_tokens for the output token count of each request "
        "(must also set --limit-max-tokens). "
        "Overrides output token count from the input dataset. "
        "Use a negative value to disable this limit.",
    )

    parser.add_argument(
        "--request-rate",
        type=float,
        default=0,
        help="Expected request rate (Poisson process) per client in requests/sec."
        "Set to 0 for no delay between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MULTITURN_BENCH_MAX_RETRIES", "0")),
        help="Maximum number of retry attempts for timed-out requests. "
        "Default is 0 (no retries). "
        "Set to higher values to retry failed requests and maintain "
        "fair workload distribution. "
        "Can also be set via MULTITURN_BENCH_MAX_RETRIES environment variable.",
    )
    parser.add_argument(
        "--conversation-sampling",
        type=ConversationSampling,
        choices=list(ConversationSampling),
        default=ConversationSampling.ROUND_ROBIN,
        help=(
            "Strategy for selecting which conversation to use for the next request. "
            "Options: 'round_robin' (cycle through conversations), "
            "'random' (pick randomly)."
        ),
    )
    parser.add_argument(
        "--verify-output",
        default=False,
        action="store_true",
        help="Verify the LLM output (compare to the answers in the input JSON file)",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=120,
        help="Timeout in seconds for each API request (default: 120). "
        "Automatically increased if max tokens imply longer decoding.",
    )

    parser.add_argument(
        "--no-stream",
        default=False,
        action="store_true",
        help="Disable stream/streaming mode (set 'stream' to False in the API request)",
    )

    parser.add_argument(
        "-e",
        "--excel-output",
        default=False,
        action="store_true",
        help="Export summary to Excel file (optional)",
    )
    parser.add_argument(
        "--stats-json-output",
        type=str,
        default=None,
        help="Export per-request stats (ttft_ms, tpot_ms, etc.) to a JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--print-content",
        default=False,
        action="store_true",
        help="Print the user prompts and the server's answers",
    )

    parser.add_argument(
        "--warmup-percentages",
        type=str,
        default="0%",
        help="Ignore the first X samples as warmup (X is a percentage)."
        " A comma separated list of percentages can be used "
        "(for example: --warmup-percentages=0%%,50%%)",
    )

    args = parser.parse_args()

    logger.info(args)

    logger.info(f"{Color.GREEN}Input parameters:{Color.RESET}")
    logger.info(f"url={args.url}")
    logger.info(f"model={args.model}")
    logger.info(f"num_clients={args.num_clients}")

    if args.verify_output:
        logger.info(f"{Color.PURPLE}Verify is enabled{Color.RESET}")

    try:
        warmup_percentages: list[float] = [0.0]
        if not args.warmup_step:
            warmup_strings: list[str] = args.warmup_percentages.split(",")
            warmup_strings = [x.replace("%", "") for x in warmup_strings]
            warmup_percentages = [float(x) / 100 for x in warmup_strings]

            for p in warmup_percentages:
                assert p >= 0.0 and p < 1.0

            warmup_percentages.sort()

            logger.info(
                f"Warmup percentages (percentage of samples): {warmup_percentages}"
            )

    except Exception:
        raise ValueError(
            f"Invalid --warmup-percentage={args.warmup_percentage}"
        ) from None

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    await get_server_info(args.url)

    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file) as f:
        input_data = json.load(f)

    gen_conv_args = None
    if isinstance(input_data, list):
        logger.info(f"Found {len(input_data)} items in the input file")

        conversations = conversations_list_to_dict(input_data)

    elif isinstance(input_data, dict):
        if "filetype" not in input_data:
            raise Exception(
                f"Input file {args.input_file} is invalid (missing 'filetype')"
            )

        logger.info(f"Using input file with filetype: {input_data['filetype']}")

        gen_conv_args = parse_input_json_file(input_data)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        conversations = generate_conversations(gen_conv_args, tokenizer)

    else:
        raise Exception(f"Input file {args.input_file} is invalid")

    if args.max_turns is not None:
        if args.max_turns < 1:
            raise ValueError("Max turns must be a positive number")
        logger.info(
            f"{Color.PURPLE}Max turns per conversation "
            f"is limited to {args.max_turns}{Color.RESET}"
        )

    client_args, req_args = get_client_config(args, conversations)

    bench_args = BenchmarkArgs(
        url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop
    )

    warmup_runtime_sec: float | None = None

    if args.warmup_step:
        warmup_client_args = client_args._replace(
            skip_first_turn=False, max_turns=1, max_active_conversations=1
        )

        warmup_bench_args = bench_args._replace(early_stop=False)

        logger.info("%sWarmup start%s", Color.PURPLE, Color.RESET)
        warmup_start_ns = time.perf_counter_ns()
        conversations, _ = await main_mp(
            warmup_client_args, req_args, warmup_bench_args, tokenizer, conversations
        )
        warmup_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - warmup_start_ns)
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.PURPLE,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info("%sWarmup done%s", Color.PURPLE, Color.RESET)

    benchmark_start_ns = time.perf_counter_ns()
    client_convs, client_metrics = await main_mp(
        client_args, req_args, bench_args, tokenizer, conversations
    )
    benchmark_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - benchmark_start_ns)

    requests_per_sec = len(client_metrics) / benchmark_runtime_sec
    benchmark_runtime_ms = benchmark_runtime_sec * 1000.0
    logger.info(
        "%sAll clients finished, benchmark runtime: %.3f sec (%.3f ms), "
        "requests per second: %.3f%s",
        Color.GREEN,
        benchmark_runtime_sec,
        benchmark_runtime_ms,
        requests_per_sec,
        Color.RESET,
    )
    if warmup_runtime_sec is not None:
        total_runtime_sec = benchmark_runtime_sec + warmup_runtime_sec
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.GREEN,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info(
            "%sTotal runtime (including warmup): %.3f sec (%.3f ms)%s",
            Color.GREEN,
            total_runtime_sec,
            total_runtime_sec * 1000,
            Color.RESET,
        )

    params = {
        "model": args.model,
        "num_clients": args.num_clients,
        "num_conversations": len(conversations),
        "active_conversations": args.max_active_conversations,
        "seed": args.seed,
    }

    if args.limit_min_tokens > 0:
        params["min_tokens"] = args.limit_min_tokens

    if args.limit_max_tokens > 0:
        params["max_tokens"] = args.limit_max_tokens

    process_statistics(
        client_metrics,
        test_params=params,
        warmup_percentages=warmup_percentages,
        verbose=args.verbose,
        gen_conv_args=gen_conv_args,
        excel_output=args.excel_output,
        warmup_runtime_sec=warmup_runtime_sec,
    )

    if args.stats_json_output is not None:
        stats_data = [s._asdict() for s in client_metrics]
        logger.info(
            f"{Color.GREEN}Writing per-request stats JSON: "
            f"{args.stats_json_output}{Color.RESET}"
        )
        os.makedirs(
            os.path.dirname(os.path.abspath(args.stats_json_output)), exist_ok=True
        )
        with open(args.stats_json_output, "w") as f:
            json.dump(stats_data, f, indent=2)

    if args.output_file is not None:
        output_data: ShareGptConversations = conversations_dict_to_list(client_convs)
        logger.info(
            f"{Color.GREEN}Writing conversations file: {args.output_file}{Color.RESET}"
        )
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
