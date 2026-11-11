from gemini_client import GeminiTrafficClient


def test_mock_result_is_flagged_as_simulated():
    client = GeminiTrafficClient(api_key="")
    result = client._mock_rider_safety_analysis()
    assert getattr(result, "_simulated", False) is True
    assert "SIMULASI" in result.officer_notes
    assert "_simulated" not in result.model_dump()
