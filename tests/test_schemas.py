from pydantic import ValidationError
import pytest

from src.schemas import (
    PropertyInput,
    PriceResponse,
    HealthResponse,
    ModelInfoResponse,
)


def test_property_input_valid():
    property_data = PropertyInput(
        property_type=2,
        area=1200,
        sub_area="kothrud",
        clubhouse=1,
        school=1,
        hospital=0,
        mall=0,
        park=1,
        pool=0,
        gym=1,
    )

    assert property_data.property_type == 2
    assert property_data.area == 1200
    assert property_data.sub_area == "kothrud"
    assert property_data.description == ""


def test_property_input_requires_required_fields():
    with pytest.raises(ValidationError):
        PropertyInput(
            property_type=2,
            area=1200,
            sub_area="kothrud",
        )


def test_price_response():
    response = PriceResponse(
        predicted_price=125.5,
        lower_bound=110.0,
        upper_bound=140.0,
        features_used=25,
    )

    assert response.predicted_price == 125.5
    assert response.lower_bound == 110.0
    assert response.upper_bound == 140.0
    assert response.features_used == 25


def test_health_response():
    response = HealthResponse(status="API is healthy and running.")

    assert response.status == "API is healthy and running."


def test_model_info_response():
    response = ModelInfoResponse(
        model_type="VotingRegressor",
        vectorizer_vocab_size=5000,
        interval_margin=15.5,
    )

    assert response.model_type == "VotingRegressor"
    assert response.vectorizer_vocab_size == 5000
    assert response.interval_margin == 15.5
