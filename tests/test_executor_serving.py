"""Development endpoint contract checks, with no model loading or network."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from training.integration_smoke import assess_fixture_answer
from training.serve_executor import ChatRequest, create_app


def test_integration_answer_rejects_duplicate_products_and_wrong_price() -> None:
    good = "商品 CUP-OK，标价 20 欧元，不是到手总价。"
    assert all(assess_fixture_answer(good).values())
    duplicate = "候选 CUP-OK（标价20欧元）和 CUP-OK（标价20欧元），不是到手总价。"
    assert not all(assess_fixture_answer(duplicate).values())
    assert not all(assess_fixture_answer(good.replace("20 欧元", "120 欧元")).values())
    assert not all(
        assess_fixture_answer(good.replace("不是到手总价", "含税包邮")).values()
    )
    envelope = '{"results":[{"task":"说明不是到手总价", "answer":"CUP-OK，20欧元"}]}'
    assert not assess_fixture_answer(envelope)["price_is_not_checkout_total"]


def test_local_endpoint_rejects_unsupported_requests_before_generation() -> None:
    # Not entering lifespan: all rejected requests must be validated before any
    # model access. No downloaded weights are necessary for this contract check.
    client = TestClient(create_app("unused", None, "executor-v1"))
    base = {"model": "executor-v1", "messages": [{"role": "user", "content": "hello"}]}
    assert (
        client.post("/v1/chat/completions", json={**base, "model": "wrong"}).status_code
        == 404
    )
    assert (
        client.post("/v1/chat/completions", json={**base, "stream": True}).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/chat/completions", json={**base, "tool_choice": "required"}
        ).status_code
        == 400
    )
    assert client.post("/v1/chat/completions", json={**base, "n": 2}).status_code == 400
    assert (
        client.post("/v1/chat/completions", json={**base, "messages": []}).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/chat/completions",
            json={
                **base,
                "messages": [{"role": "user", "content": [{"type": "image_url"}]}],
            },
        ).status_code
        == 400
    )
    assert (
        client.post("/v1/chat/completions", json={**base, "max_tokens": 0}).status_code
        == 422
    )


def test_actual_chat_client_payload_matches_local_server_schema():
    seen = []

    def transport(request):
        payload = ChatRequest.model_validate(json.loads(request.content))
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "contract-test",
                "object": "chat.completion",
                "created": 0,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "contract accepted",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            model = ChatOpenAI(
                model="contract-only",
                base_url="http://127.0.0.1:8010/v1",
                api_key="local-no-key",
                temperature=0,
                max_tokens=256,
                max_retries=0,
                http_async_client=client,
            )
            answer = await model.bind_tools([create_item_search_tool(None)]).ainvoke(
                "查询商品"
            )
            assert answer.text == "contract accepted"

    asyncio.run(scenario())
    assert len(seen) == 1
    assert seen[0].tools[0]["function"]["name"] == "item_search"
