# GEMINI.md

Este proyecto es un pipeline de pronóstico de series temporales probabilístico, basado en **FEDformer** y **Normalizing Flows**, optimizado para entornos de investigación financiera (predicción de volatilidad, VaR/CVaR, backtesting walk-forward).

## Visión General del Proyecto
El sistema permite realizar pronósticos con bandas de incertidumbre, evaluación probabilística y simulación de riesgos. Incluye herramientas para backtesting con validación walk-forward anti-fuga (anti-leakage), búsqueda de hiperparámetros con Optuna, y exportación de modelos especialistas.

- **Tecnologías Principales:** Python 3.10+, PyTorch, Pandas, NumPy, YFinance (ingesta unificada), Optuna (optimización).
- **Arquitectura:** Flujo desde `main.py` -> `WalkForwardTrainer` -> `Flow_FEDformer`.

## Construcción y Ejecución
Para comenzar, asegúrate de activar el entorno virtual y tener las dependencias instaladas:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Comandos Clave
*   **Entrenamiento (Canonical run):**
    ```bash
    MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" ...
    ```
*   **Inferencia (Especialista guardado):**
    ```bash
    python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --plot
    ```
*   **Búsqueda de Hiperparámetros (Optuna):**
    ```bash
    python3 tune_hyperparams.py --csv data/NVDA_features.csv --n-trials 8 ...
    ```

## Convenciones de Desarrollo
*   **Gestión de Datos:** Los datos se procesan mediante `financial_dataset_builder` (fuente unificada en `yfinance`). Todo dataset debe cumplir con el contrato de 11 características (OHLCV + VIX + TAs).
*   **Testing:** Se utiliza `pytest`. Todo cambio en módulos core (`data/`, `models/`, `training/`) debe ir acompañado de pruebas unitarias o de integración.
*   **Integridad:** Se ha implementado `strict=True` por defecto para asegurar que los fallos en la ingesta de datos detengan la ejecución (`fail-fast`).
*   **Calidad:** Se utilizan herramientas de linting y formateo (Ruff, Pylint). Ejecutar `make ci-check` antes de cualquier PR.

## Notas para el Agente
*   **Módulo `data`:** Este módulo centraliza la ingesta (consolidada en `yfinance`) y la preparación de datos. Está estrictamente ligado a `config.py` y `FEDformerConfig`.
*   **Configuración:** Evitar mutar el objeto `FEDformerConfig` directamente durante la ejecución. Los parámetros derivados deben calcularse de forma externa y pasarse explícitamente.
*   **Seguridad:** No exponer credenciales de API (ej. Alpha Vantage si se vuelve a habilitar) en commits. Actualmente el sistema prioriza fuentes públicas (yfinance).
