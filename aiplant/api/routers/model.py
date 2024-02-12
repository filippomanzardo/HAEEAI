from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from aiplant.bluetooth.adapter import BluetoothAdapter
from aiplant.database.eeprom import EEPROMDatabase
from aiplant.model.labeler import Labeler
from aiplant.model.models import Feature
from aiplant.model.waterer import Waterer


def create_model_router(
    waterer: Waterer,
    labeler: Labeler,
    database: EEPROMDatabase,
    adapter: BluetoothAdapter,
) -> APIRouter:
    """Build a FastAPI router for serving model endpoints."""
    router = APIRouter()

    @router.post("/train")
    async def train_waterer(background_tasks: BackgroundTasks) -> JSONResponse:
        """Train the waterer model."""
        # We asynchronously train the model in the background.
        background_tasks.add_task(waterer.train)

        return JSONResponse(
            content={"message": "Training the waterer model."},
        )

    @router.get("/tflite/test")
    async def test_tflite() -> JSONResponse:
        """Test the TFLite model."""
        result = await waterer.test_tf_lite_accuracy()

        return JSONResponse(
            content={"message": "Testing the TFLite model.", "result": result},
        )

    @router.get("/label")
    async def label_features() -> JSONResponse:
        """Label the feature."""
        database.log_status()
        result = await labeler.label_samples(database.real_time_samples)

        database.load_samples(result)

        parsed_result = [
            {
                "timestamp": feature.timestamp,
                "moisture": feature.moisture,
                "temperature": feature.temperature,
                "label": bool(label),
            }
            for feature, label in result
        ]

        return JSONResponse(
            content={
                "message": "Labeling the feature.",
                "result": jsonable_encoder(parsed_result),
            },
        )

    @router.post("/send-new-model")
    async def send_new_model(background_tasks: BackgroundTasks) -> JSONResponse:
        """Send the new model."""
        model = await waterer.convert_to_tflite()

        background_tasks.add_task(adapter.send_data, bytearray(to_hex_array(model)))

        return JSONResponse(
            content={"message": "Sending the new model to AIPlant."},
        )

    @router.post("/predict")
    async def predict_target(feature: dict[str, Any]) -> JSONResponse:
        """Predict the target."""
        target = await waterer.predict(Feature(**feature))
        return JSONResponse(
            content={"message": "Predicting the target.", "target": target},
        )

    return router


def to_hex_array(tflite_model: bytes) -> list[int]:
    """Convert the data to a hex array."""
    hex_array = []
    for i, val in enumerate(tflite_model):
        # Construct string from hex
        hex_str = format(val, "#04x")
        # Add formatting so each line stays within 80 characters
        hex_array.append(int(hex_str, 0))

    return hex_array
