"""Protocol errors and single-operation credit budgets, without provider calls."""
import json
import unittest
from unittest.mock import AsyncMock, patch
from mcp.client import Client
from youtube_translate_mcp import server


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_url_is_an_actual_mcp_tool_error(self):
        async with Client(server.mcp) as client:
            result = await client.call_tool('get_transcript', {'url': 'not-a-youtube-url'})
        self.assertTrue(result.is_error)
        self.assertIn('supported YouTube URL', result.content[0].text)

    async def test_quote_never_submits_a_processing_job(self):
        with patch.object(server, 'api_request', AsyncMock(return_value={'quote': {'maximumCreditUnits': 300000}})) as request:
            result = await server.get_processing_quote('https://www.youtube.com/watch?v=Ro6MDU2CFO4')
        self.assertEqual(json.loads(result)['quote']['maximumCreditUnits'], 300000)
        request.assert_awaited_once()
        self.assertEqual(request.await_args.args, ('POST', '/api/processing/quote'))

    async def test_start_video_forwards_the_explicit_transcription_budget(self):
        with patch.object(server, 'api_request', AsyncMock(return_value={'id': 'owner-job', 'status': 'processing'})) as request:
            await server.start_video('https://www.youtube.com/watch?v=Ro6MDU2CFO4', 300000)
        self.assertEqual(request.await_args.kwargs['json_body']['max_credit_units'], 300000)

    async def test_summary_budget_is_forwarded_to_the_transcription_step(self):
        calls = []

        async def api(method, route, **kwargs):
            calls.append((method, route, kwargs))
            if route == '/api/videos':
                return {'id': 'owner-video', 'status': 'completed'}
            if route.endswith('/summarize/status?language=en&length=short'):
                return {'data': {'status': 'completed'}}
            if route.endswith('/summary'):
                return {'data': {'summary': 'Owner test summary'}}
            return {'data': {'status': 'processing'}}

        with patch.object(server, 'api_request', api):
            self.assertEqual(await server.get_summary('https://www.youtube.com/watch?v=Ro6MDU2CFO4', 'en', 'short', 5000), 'Owner test summary')
        self.assertEqual(calls[0][2]['json_body']['max_credit_units'], 5000)
        summary = next(call for call in calls if call[1].endswith('/summarize'))
        self.assertEqual(summary[2]['json_body']['max_credit_units'], 5000)


if __name__ == '__main__':
    unittest.main()
