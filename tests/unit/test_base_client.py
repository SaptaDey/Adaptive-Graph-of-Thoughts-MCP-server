# Additional tests for BaseClient

def test_invalid_constructor_args():
    # base_url must be a string, max_retries non-negative, timeout positive
    with pytest.raises(TypeError):
        BaseClient(base_url=None)
    with pytest.raises(ValueError):
        BaseClient(base_url="https://api.example.com", max_retries=-1)
    with pytest.raises(ValueError):
        BaseClient(base_url="https://api.example.com", timeout=0)


def test_non_string_endpoint(client):
    # endpoint must be a string
    with pytest.raises(TypeError):
        client.send("GET", 123)


def test_send_with_none_payload(client):
    # sending with a None payload should include json=None
    with patch("myproject.base_client.requests.Session") as mock_session:
        mock_sess = mock_session.return_value
        mock_resp = MagicMock(status_code=200, json=lambda: {"received": None})
        mock_sess.request.return_value = mock_resp

        resp = client.send("POST", "/data", None)
        assert resp == {"received": None}
        mock_sess.request.assert_called_once_with(
            "POST",
            "https://api.example.com/data",
            timeout=2,
            json=None
        )


def test_close_idempotent(client):
    # calling close() multiple times only closes the session once
    with patch("myproject.base_client.requests.Session") as mock_session:
        mock_sess = mock_session.return_value

        client.close()
        client.close()

        mock_sess.close.assert_called_once()