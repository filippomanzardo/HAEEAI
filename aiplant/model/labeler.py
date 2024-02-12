import asyncio
import logging

from sklearn.svm import SVC

from aiplant.model.models import Feature, Sample

_LOGGER = logging.getLogger(__name__)


class Labeler:
    """A class to represent the Labeler."""

    def __init__(
        self, base_samples: list[Sample], *, _model: SVC | None = None
    ) -> None:
        self._base_samples = base_samples
        self._additional_samples = list[Sample]()
        self._model = _model or SVC()

    def add_base_samples(self, samples: list[Sample]) -> None:
        """Add base samples."""
        self._base_samples.extend(samples)

    async def refresh_model(self) -> None:
        """Refresh the model."""
        combined_samples = self._base_samples + self._additional_samples
        features = [sample[0] for sample in combined_samples]
        labels = [sample[1] for sample in combined_samples]

        training_features = [
            (feature.temperature, feature.moisture) for feature in features
        ]

        _LOGGER.info("🌱 Training Labeler with %s samples 🌱", len(training_features))

        def _train() -> None:
            self._model.fit(training_features, labels)

        await asyncio.to_thread(_train)

        training_accuracy = self._model.score(training_features, labels)
        _LOGGER.info("🎯 Training Labeler accuracy: %s 🎯", training_accuracy)
        _LOGGER.info("🌱 Labeler trained 🌱")

    def label(self, feature: Feature) -> bool:
        """Label the feature."""
        return self._model.predict([(feature.temperature, feature.moisture)])[0]

    async def label_samples(self, features: list[Feature]) -> list[Sample]:
        """Label the samples."""
        samples = [Sample((feature, self.label(feature))) for feature in features]
        self._additional_samples.extend(samples)

        return samples
