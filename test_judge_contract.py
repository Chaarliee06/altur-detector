"""Development checks for fallback semantics; official public client is the gate."""
import base64
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as service
from check_endpoint import wav_bytes


class JudgeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(service.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def assert_abstention(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"is_synthetic": False, "confidence": .5})

    def test_errors_and_missing_audio_always_abstain(self):
        for payload in ({}, {"call_id": "x", "sample_rate": "ignored", "channels": []},
                        {"audio_base64": None}, {"audio": None}, {"audio_base64": "%%%"},
                        {"audio_base64": 3}, {"audio": []}, [], None,
                        {"audio": base64.b64encode(b"not a WAV").decode()}):
            with self.subTest(payload=payload):
                self.assert_abstention(self.client.post("/detect", json=payload))
        self.assert_abstention(self.client.post("/detect", content="{", headers={"Content-Type": "application/json"}))
        self.assert_abstention(self.client.get("/detect"))

    def test_both_fields_and_ignored_metadata(self):
        audio = base64.b64encode(wav_bytes(pattern="many")).decode()
        old = self.client.post("/detect", json={"audio": audio, "format": "wav"})
        new = self.client.post("/detect", json={"audio_base64": audio, "call_id": "fixture", "sample_rate": "ignored", "channels": {"ignored": True}})
        self.assertEqual(old.status_code, 200)
        self.assertEqual(new.json(), old.json())
        self.assertIs(type(new.json()["is_synthetic"]), bool)
        self.assertEqual(self.client.post("/detect/", json={"audio_base64": "", "audio": audio}).json(), old.json())

    def test_inference_exception_and_body_limit(self):
        with patch.object(service, "classify", side_effect=RuntimeError("fixture")):
            self.assert_abstention(self.client.post("/detect", json={"audio_base64": "anything"}))
        with patch.object(service, "MAX_REQUEST_BYTES", 16):
            self.assert_abstention(self.client.post("/detect", json={"audio_base64": "x" * 32}))

    def test_cors_on_normal_fallback_and_preflight(self):
        origin = {"Origin": "https://example.com"}
        response = self.client.post("/detect", json={}, headers=origin)
        self.assert_abstention(response)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        response = self.client.options("/detect", headers={**origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")


if __name__ == "__main__":
    unittest.main()
