import asyncio
import logging
from datetime import datetime, timezone
from typing import Final

from bleak import BleakClient, BleakGATTCharacteristic, BleakScanner, BLEDevice
from bleak.backends.service import BleakGATTService

from aiplant.bluetooth.exceptions import CharacteristicNotFound
from aiplant.model.models import Feature

_LOGGER = logging.getLogger(__name__)

_DEVICE_NAME: Final[str] = "AIPlant"
_READ_TYPE: Final[str] = "read"
_WRITE_TYPE: Final[str] = "write"
_MOISTURE_CHARACTERISTIC_UUID_PREFIX: Final[str] = "00001000"
_TEMPERATURE_CHARACTERISTIC_UUID_PREFIX: Final[str] = "00002000"
_MODEL_CHARACTERISTIC_UUID_PREFIX: Final[str] = "00003000"
_END_SEQUENCE: Final[bytearray] = bytearray("END__OF__MODEL__SEQUENCE".encode("utf-8"))


class BluetoothAdapter:
    def __init__(self, device_name: str = _DEVICE_NAME) -> None:
        self._device_name = device_name
        self._device: BLEDevice | None = None
        self._client: BleakClient | None = None
        self._read_enabled = True

    @property
    def device(self) -> BLEDevice:
        if self._device is None:
            raise RuntimeError("Device wasn't initialized.")
        return self._device

    @property
    def client(self) -> BleakClient:
        if self._client is None:
            raise RuntimeError("Client wasn't initialized.")
        return self._client

    @property
    def ai_plant_service(self) -> BleakGATTService:
        return next(iter(self.client.services))

    @property
    def moisture_sensor(self) -> BleakGATTCharacteristic:
        characteristic = next(
            (
                characteristic
                for characteristic in self.ai_plant_service.characteristics
                if _READ_TYPE in characteristic.properties
                and characteristic.uuid.startswith(_MOISTURE_CHARACTERISTIC_UUID_PREFIX)
            ),
            None,
        )

        if characteristic is None:
            raise CharacteristicNotFound(
                f"Characteristic with UUID prefix {_MOISTURE_CHARACTERISTIC_UUID_PREFIX} not found."
            )

        return characteristic

    @property
    def temperature_sensor(self) -> BleakGATTCharacteristic:
        characteristic = next(
            (
                characteristic
                for characteristic in self.ai_plant_service.characteristics
                if _READ_TYPE in characteristic.properties
                and characteristic.uuid.startswith(
                    _TEMPERATURE_CHARACTERISTIC_UUID_PREFIX
                )
            ),
            None,
        )

        if characteristic is None:
            raise CharacteristicNotFound(
                f"Characteristic with UUID prefix {_TEMPERATURE_CHARACTERISTIC_UUID_PREFIX} not found."
            )

        return characteristic

    @property
    def model_characteristic(self) -> BleakGATTCharacteristic:
        characteristic = next(
            (
                characteristic
                for characteristic in self.ai_plant_service.characteristics
                if _WRITE_TYPE in characteristic.properties
                and characteristic.uuid.startswith(_MODEL_CHARACTERISTIC_UUID_PREFIX)
            ),
            None,
        )

        if characteristic is None:
            raise CharacteristicNotFound(
                f"Characteristic with UUID prefix {_MODEL_CHARACTERISTIC_UUID_PREFIX} not found."
            )

        return characteristic

    async def connect(self) -> None:
        """Connect to the Bluetooth adapter."""
        device_found = False
        while not device_found:
            available_devices = await BleakScanner.discover()
            _LOGGER.info(
                f"Available devices: {[device.name for device in available_devices if device.name]}"
            )
            for device in available_devices:
                if self._device_name == device.name:
                    self._device = device
                    device_found = True
                    _LOGGER.info(f"Device {self._device_name} found.")

            if not device_found:
                _LOGGER.info(f"Device {self._device_name} not found. Retrying...")

        await self.initialize_client()

    async def initialize_client(self) -> None:
        """Initialize the Bluetooth client."""
        if self._device is None:
            raise RuntimeError("Device wasn't initialized.")
        self._client = BleakClient(self._device)
        await self._client.connect()

        for characteristic in self.ai_plant_service.characteristics:
            _LOGGER.info(f"Characteristic: {characteristic.uuid}")

    async def disconnect(self) -> None:
        """Disconnect from the Bluetooth adapter."""
        await self.client.disconnect()

    async def read_data(self) -> tuple[bytearray, bytearray]:
        """Read data from the Bluetooth adapter."""
        while not self._read_enabled:
            await asyncio.sleep(1)

        moisture, temperature = await asyncio.gather(
            self.client.read_gatt_char(self.moisture_sensor),
            self.client.read_gatt_char(self.temperature_sensor),
        )

        return moisture, temperature

    async def get_feature(self) -> Feature:
        """Get the features of the Bluetooth adapter."""
        moisture, temperature = await self.read_data()
        timestamp = datetime.now(tz=timezone.utc)
        return Feature(
            timestamp=int(timestamp.timestamp()),
            temperature=float(temperature.decode("utf-8")),
            moisture=float(moisture.decode("utf-8")),
        )

    async def send_data(self, data: bytearray) -> None:
        """Send data to the Bluetooth adapter."""
        self._read_enabled = False
        _LOGGER.info(
            "⚙️ About to send data %s to the model characteristic %s ⚙️",
            len(data),
            self.model_characteristic.uuid,
        )
        if len(data) > 512:
            _LOGGER.info("⚙️ Data is too big. Splitting it into chunks... ⚙️")
            for chunk in chunk_data(data, 512):
                _LOGGER.info("⚙️ Sending chunk %s ⚙️", chunk)
                await self.client.write_gatt_char(
                    self.model_characteristic, chunk, response=True
                )
                await asyncio.sleep(2)

            await self.client.write_gatt_char(
                self.model_characteristic, _END_SEQUENCE, response=True
            )
        else:
            await self.client.write_gatt_char(
                self.model_characteristic, data, response=True
            )

        self._read_enabled = True


def chunk_data(data: bytearray, size: int) -> list[bytearray]:
    """Chunk the data into smaller chunks."""
    return [data[i : i + size] for i in range(0, len(data), size)]
