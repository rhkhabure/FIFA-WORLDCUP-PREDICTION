import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from security import allowed_hosts, require_admin


def _request(headers=None):
    encoded_headers = [
        (name.lower().encode(), value.encode())
        for name, value in (headers or {}).items()
    ]
    return Request({"type": "http", "headers": encoded_headers})


class SecurityTests(unittest.TestCase):
    def test_allowed_hosts_use_local_defaults_when_unconfigured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOWED_HOSTS", None)
            self.assertEqual(
                allowed_hosts(), ["localhost", "127.0.0.1", "testserver"]
            )

    def test_allowed_hosts_parse_deployment_configuration(self):
        with patch.dict(
            os.environ,
            {"ALLOWED_HOSTS": "example.com, api.example.com "},
            clear=False,
        ):
            self.assertEqual(allowed_hosts(), ["example.com", "api.example.com"])

    def test_admin_requires_configuration(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADMIN_API_TOKEN", None)
            with self.assertRaisesRegex(
                HTTPException, "Administrative endpoints are not configured"
            ):
                require_admin(_request())

    def test_admin_accepts_only_matching_bearer_token(self):
        with patch.dict(os.environ, {"ADMIN_API_TOKEN": "test-token"}, clear=False):
            with self.assertRaises(HTTPException) as context:
                require_admin(_request({"Authorization": "Bearer wrong-token"}))
            self.assertEqual(context.exception.status_code, 401)
            require_admin(_request({"Authorization": "Bearer test-token"}))


if __name__ == "__main__":
    unittest.main()
