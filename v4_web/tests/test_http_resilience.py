import io
import unittest
import urllib.error
import urllib.request

from http_resilience import urlopen_with_retry


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class HttpResilienceTests(unittest.TestCase):
    def test_retries_transient_transport_failure_with_backoff(self):
        attempts = []
        delays = []

        def opener(_request, **_kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise urllib.error.URLError("temporary outage")
            return _Response()

        response = urlopen_with_retry(
            urllib.request.Request("https://example.test"),
            timeout=1,
            opener=opener,
            sleeper=delays.append,
        )

        self.assertIsInstance(response, _Response)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(delays, [0.25])

    def test_does_not_retry_permanent_http_failure(self):
        attempts = []

        def opener(_request, **_kwargs):
            attempts.append(True)
            error = urllib.error.HTTPError(
                "https://example.test",
                401,
                "unauthorized",
                {},
                io.BytesIO(),
            )
            try:
                raise error
            finally:
                error.close()

        with self.assertRaises(urllib.error.HTTPError):
            urlopen_with_retry(
                urllib.request.Request("https://example.test"),
                timeout=1,
                opener=opener,
                sleeper=lambda _delay: self.fail("unexpected retry"),
            )
        self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()
