import asyncio
import logging
from typing import Any, Generator, Self, Sequence

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from aiplant.database.eeprom import EEPROMDatabase
from aiplant.database.models import _DatabaseEntry
from aiplant.model.models import Feature, Sample, Target

_LOGGER = logging.getLogger(__name__)


class Waterer:
    """A class to represent the waterer model."""

    def __init__(
        self, database: EEPROMDatabase, *, _model: tf.keras.Sequential | None = None
    ) -> None:
        self._database = database
        self._model = _model
        self._lite_model = None

    @property
    def model(self) -> tf.keras.Sequential:
        """Return the model."""
        if self._model is None:
            raise RuntimeError("Model wasn't initialized.")

        return self._model

    def set_model(self, input_data: npt.NDArray[Feature]) -> None:
        """Set the model."""
        mean, variance = np.mean(input_data, axis=0), np.var(input_data, axis=0)
        self._model = tf.keras.Sequential()
        self._model.add(
            tf.keras.layers.Normalization(
                mean=mean, variance=variance, input_shape=(2,)
            )
        )
        self._model.add(tf.keras.layers.Dense(4))
        self._model.add(tf.keras.layers.Dense(2, activation="softmax"))
        self._model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def transform(self, data: Feature) -> npt.NDArray[Feature]:
        """Transform the data."""
        return np.array([data.temperature, data.moisture])

    def preprocess(
        self, data: Sequence[_DatabaseEntry]
    ) -> tuple[
        npt.NDArray[Feature],
        npt.NDArray[Target],
        npt.NDArray[Feature],
        npt.NDArray[Target],
    ]:
        """Preprocess the data."""
        _LOGGER.info("🌱 Preprocessing the data... 🌱")

        features = np.array([(sample.temperature, sample.humidity) for sample in data])
        # Reshape the features to add a sequence length dimension
        targets = np.array([int(sample.target) for sample in data])

        # we'll use 10% of the data for validation
        split = int(0.9 * len(features))
        train_x, val_x = features[:split], features[split:]
        train_y, val_y = targets[:split], targets[split:]

        return train_x.astype(np.float32), train_y, val_x.astype(np.float32), val_y

    async def train(self) -> None:
        """Train the model."""
        data = self._database.entries
        _LOGGER.info("🌱 Training Classifier with %s samples 🌱", len(data))

        train_x, train_y, val_x, val_y = self.preprocess(data)

        self.set_model(train_x)

        def _train():
            self.model.fit(x=train_x, y=train_y, epochs=100, batch_size=16, verbose=0)

        await asyncio.to_thread(_train)

        # Save the model
        self.model.save("./model/model.keras")

        training_loss, training_accuracy = self.model.evaluate(val_x, val_y, verbose=0)
        _LOGGER.info("💪🏻 Training Classifier loss: %s 💪🏻", training_loss)
        _LOGGER.info("🎯 Training Classifier accuracy: %s 🎯", training_accuracy)

        _LOGGER.info("🌱 Classifier trained 🌱")

    async def convert_to_tflite(self) -> bytearray:
        """Convert the model to a TFLite model."""
        _LOGGER.info("🌱 Converting the model to TFLite... 🌱")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        _, _, val_x, val_y = self.preprocess(self._database.entries)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]

        self._lite_model = converter.convert()

        # Save to ./model/model.tflite
        with open("./model/model.tflite", "wb") as f:
            f.write(self._lite_model)

        with open("./model/model.h", "w") as f:
            f.write(hex_to_c_array(self._lite_model, "model"))

        return self._lite_model

    async def test_tf_lite_accuracy(self) -> Any:
        """Test the accuracy of the TFLite model."""
        if self._lite_model is None:
            await self.convert_to_tflite()

        interpreter = tf.lite.Interpreter(model_content=self._lite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        data = self._database.entries
        _LOGGER.info("🌱 Testing TFLite Classifier with %s samples... 🌱", len(data))

        _, _, val_x, val_y = self.preprocess(data)

        def _test():
            predictions = []
            total_loss = 0
            correct_samples = 0
            for sample in val_x:
                interpreter.set_tensor(
                    input_details[0]["index"], sample.reshape(1, -1).astype(np.float32)
                )

                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]["index"])
                predictions.append(prediction)

            for i, prediction in enumerate(predictions):
                correct_samples += np.argmax(predictions[0]) == val_y[i]

            accuracy = correct_samples / len(val_y)

            _LOGGER.info("💪🏻 TFLite Classifier loss: %s 💪🏻", total_loss)
            _LOGGER.info("🎯 TFLite Classifier accuracy: %s 🎯", accuracy)

            return accuracy

        accuracy = await asyncio.to_thread(_test)

        return accuracy

    async def predict(self, data: Feature) -> Target:
        """Predict the target."""
        sample = self.transform(data)

        _LOGGER.info(
            "🌱 Predicting the target (%s, %s, %s)... 🌱",
            data.timestamp,
            data.temperature,
            data.moisture,
        )
        return self.model.predict(sample).tolist()


# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data: bytes, var_name: str) -> str:
    """Function to build a C array from a bytes object."""

    c_str = ""

    # Create header guard
    c_str += "#ifndef " + var_name.upper() + "_H\n"
    c_str += "#define " + var_name.upper() + "_H\n\n"

    # Add array length at top of file
    c_str += "\nunsigned int " + var_name + "_len = " + str(len(hex_data)) + ";\n"

    # Declare C variable
    c_str += "unsigned char " + var_name + "[] = {"
    hex_array = []
    for i, val in enumerate(hex_data):
        # Construct string from hex
        hex_str = format(val, "#04x")

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ","
        if (i + 1) % 12 == 0:
            hex_str += "\n "
        hex_array.append(hex_str)

    # Add closing brace
    c_str += "\n " + format(" ".join(hex_array)) + "\n};\n\n"

    # Close out header guard
    c_str += "#endif //" + var_name.upper() + "_H"

    return c_str
