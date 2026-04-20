# -*- coding: utf-8 -*-
"""
Buffer de ensayo (Rehearsal Buffer) para continual learning en walk-forward training.

Almacena ventanas de entrenamientos previos (folds anteriores) y las reproduce
durante el entrenamiento del fold actual, mitigando el olvido catastrófico.
"""

import random
from collections import deque

import torch


class RehearsalBuffer:
    """Buffer FIFO de ventanas históricas para continual learning.

    Almacena muestras individuales extraídas de batches de folds anteriores
    y permite muestreo aleatorio para pasos de replay durante el entrenamiento.
    """

    def __init__(self, capacity: int, strategy: str = "uniform") -> None:
        self._capacity = capacity
        self._strategy = strategy  # "uniform" (única estrategia implementada)
        self._storage: deque[dict[str, torch.Tensor]] = deque(maxlen=capacity)

    def add_batch(self, batch: dict[str, torch.Tensor]) -> None:
        """Descompone el batch en muestras individuales y las añade al buffer (FIFO).

        Si se supera la capacidad, expulsa la muestra más antigua.
        """
        batch_size = next(iter(batch.values())).shape[0]
        for i in range(batch_size):
            sample = {k: v[i].detach().cpu() for k, v in batch.items()}
            self._storage.append(sample)  # deque(maxlen) hace FIFO automáticamente

    def sample(self, k: int) -> dict[str, torch.Tensor] | None:
        """Retorna k muestras aleatorias como batch o None si el buffer está vacío.

        Si k > len(buffer), devuelve todas las muestras disponibles.
        """
        if not self._storage:
            return None
        k = min(k, len(self._storage))
        pool = list(self._storage)  # O(n) una vez; evita O(n) por índice en deque
        sampled = random.sample(pool, k)
        return {key: torch.stack([s[key] for s in sampled]) for key in sampled[0]}

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        """Alias de add_batch; punto de extensión para estrategias con prioridad."""
        self.add_batch(batch)

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def is_ready(self) -> bool:
        """True si el buffer contiene al menos una muestra."""
        return len(self._storage) > 0
