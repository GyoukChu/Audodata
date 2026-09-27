import json
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from serving.check_glm53 import run
from tests.fake_openai_server import text_response, tool_call_response


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("keep_reasoning", [True, False])
def test_glm_smoke_round_trip_uses_both_aliases_and_verifies_rendered_history(reasoning_field, keep_reasoning):
    reasoning = "I should obtain the weather for Paris from the tool."
    tool_response = tool_call_response([("get_weather", {"city": "Paris", "unit": "celsius"})])
    tool_response["choices"][0]["message"][reasoning_field] = reasoning
    responses = iter([text_response("391", reasoning="multiply"), tool_response,
                      text_response("18 degrees celsius"), text_response("Jupiter")])
    chat_requests, paths = [], []
    rendered = None

    def handle(request):
        nonlocal rendered
        paths.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            chat_requests.append(body)
            return httpx.Response(200, json=next(responses))
        if request.url.path == "/tokenize":
            assistant = body["messages"][1]
            assert assistant["reasoning"] == assistant["reasoning_content"] == reasoning
            assert body["tools"][0]["function"]["name"] == "get_weather"
            # Emulate a template that reads only `reasoning`, as GLM does.
            rendered = assistant["reasoning"] if keep_reasoning else "history without thoughts"
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        if request.url.path == "/detokenize":
            assert body == {"model": "glm-test", "tokens": [1, 2, 3]}
            return httpx.Response(200, json={"prompt": f"<assistant>{rendered}</assistant>"})
        pytest.fail(f"unexpected request: {request.url}")

    with OpenAI(base_url="http://fake/v1", api_key="EMPTY", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        results = run(client, "glm-test", 16384)
    assert results["e_rendered_reasoning"] is keep_reasoning
    assert all(value for key, value in results.items() if key != "e_rendered_reasoning")
    assistant = chat_requests[2]["messages"][1]
    assert assistant["reasoning"] == assistant["reasoning_content"] == reasoning
    assert paths == ["/v1/chat/completions", "/v1/chat/completions", "/tokenize", "/detokenize",
                     "/v1/chat/completions", "/v1/chat/completions"]


@pytest.mark.parametrize("has_tool", [False, True])
def test_glm_smoke_cannot_pass_rendering_check_without_tool_reasoning(has_tool):
    response = (tool_call_response([("get_weather", {"city": "Paris", "unit": "celsius"})])
                if has_tool else text_response("no tool"))
    responses = [text_response("391", reasoning="multiply"), response]
    if has_tool:
        responses.append(text_response("18 degrees celsius"))
    responses.append(text_response("Jupiter"))
    responses = iter(responses)

    with OpenAI(base_url="http://fake/v1", api_key="EMPTY", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json=next(responses))))) as client:
        results = run(client, "glm-test", 16384)
    assert not results["e_rendered_reasoning"]


@pytest.mark.parametrize("filename", ["README.md", "docs/RUNBOOK.md"])
def test_quickstart_readiness_sequence_and_cohort_documentation(filename):
    text = (Path(__file__).resolve().parents[1] / filename).read_text()
    glm = text.index("bash serving/serve_glm53.sh")
    strong = text.index("bash serving/serve_qwen27b.sh")
    weak = text.index("bash serving/serve_qwen4b.sh")
    pipeline = text.index("autodata-run-cs --config", weak)
    assert "8000/v1/models" in text[glm:strong] and "sleep 15" in text[glm:strong]
    assert "8001/v1/models" in text[strong:weak] and "sleep 15" in text[strong:weak]
    assert "bash scripts/wait_servers.sh" in text[weak:pipeline]
    assert "cohort.json" in text and "--allow-config-mismatch" in text
