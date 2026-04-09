import json
from unittest.mock import MagicMock, patch

from atlas_rl_copilot.minimax_client import minimax_chat


def test_minimax_chat_parses_message_content() -> None:
    payload = {
        "choices": [{"message": {"content": "## Hello\n- ok", "role": "assistant"}}],
        "base_resp": {"status_code": 0, "status_msg": ""},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = None

    with patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}, clear=False):
        with patch("atlas_rl_copilot.minimax_client.urllib.request.urlopen", return_value=mock_cm):
            out = minimax_chat(system_text="sys", user_text="user")
    assert "## Hello" in out
