# GitHub Actions Workflows

Este directorio contiene los workflows automáticos de GitHub Actions para validación continua del proyecto FEDformer.

## 📋 Workflows Disponibles

### 1. **critical-fixes.yml** - Validación de Correcciones Críticas
Valida que las 5 correcciones críticas estén implementadas correctamente.

**Triggered on:**
- Push to `main` o `develop`
- Pull requests a `main` o `develop`

**Validation Steps:**
1. ✅ Walk-forward data leakage fix (trainer.py)
2. ✅ RegimeDetector volatility fix (dataset.py)
3. ✅ Fourier attention determinism (layers.py)
4. ✅ Trend projection validation (fedformer.py)
5. ✅ Log-prob scaling normalization (flows.py)
6. ✅ Regression test (core classes intact)

**Python Versions Tested:**
- Python 3.9
- Python 3.10
- Python 3.11

**Output:**
```
═══════════════════════════════════════════════════════
CRITICAL FIXES VALIDATION SUMMARY
═══════════════════════════════════════════════════════
✅ Walk-forward Data Leakage Fix: VERIFIED
✅ RegimeDetector Volatility Fix: VERIFIED
✅ Fourier Attention Determinism: VERIFIED
✅ Trend Projection Validation: VERIFIED
✅ Log-Prob Scaling Normalization: VERIFIED
✅ Regression Test: NO REGRESSIONS DETECTED
═══════════════════════════════════════════════════════
```

---

### 2. **compatibility.yml** - Tests de Compatibilidad
Verifica la compatibilidad del proyecto con diferentes versiones de Python y valida que todas las integraciones funcionen correctamente.

**Triggered on:**
- Push to `main` o `develop`
- Pull requests a `main` o `develop`

**Compatibility Checks:**
1. 📦 Module imports verification
2. ⚙️ FEDformerConfig initialization
3. 📊 RegimeDetector with volatility fix
4. 🎯 Fourier Attention determinism
5. 🔄 Flow_FEDformer forward pass
6. 🌊 NormalizingFlow log-prob scaling
7. ✔️ Configuration validation
8. 🔗 No breaking changes detection

**Python Versions:**
- Python 3.9
- Python 3.10
- Python 3.11

**Key Tests:**
- Module imports (todos los módulos principales)
- Configuración por defecto
- Regimen detection con datos aleatorios
- Forward pass del modelo
- Stabilidad numérica del flow
- Backward compatibility

---

### 3. **security.yml** - Análisis de Seguridad y Calidad
Analiza seguridad del código, calidad del código y verifica la integridad de las correcciones críticas.

**Triggered on:**
- Push a `main` o `develop`
- Pull requests a `main` o `develop`
- Schedule: Semanalmente (domingos a las 00:00 UTC)

**Quality Checks:**
- 🔍 Linting con flake8
- 📐 Formato con black
- 📑 Orden de imports con isort
- 🔐 Scanning de seguridad
- 📦 Análisis de dependencias

**Security Checks:**
- Detección de credenciales hardcodeadas
- Detección de eval/exec/pickle.loads
- Patrones potenciales de SQL injection
- Verificación de dependencias vulnerables
- Integridad de correcciones críticas

**Output Sections:**
```
Code Quality:
- ✅ Black formatting check
- ✅ Import order verification
- ✅ Flake8 linting
- ✅ Security scanning

Dependencies:
- ✅ Requirements.txt analysis
- ✅ Core dependency verification

Fixes Integrity:
- ✅ All 5 critical fixes present
- ✅ Signatures verified
```

---

## 🚀 Cómo Funcionan los Workflows

### Ejecución Manual
Todos los workflows se ejecutan automáticamente en:
1. **Cada push** a las ramas `main` o `develop`
2. **Cada Pull Request** a las ramas `main` o `develop`
3. **Semanalmente** (security.yml) - Domingo a las 00:00 UTC

### Monitoreo
Para ver el estado de los workflows:
1. Ve al repositorio en GitHub
2. Click en **Actions** tab
3. Selecciona el workflow que quieras ver
4. Haz click en la ejecución más reciente para detalles

### Resultados
- ✅ **Exitoso**: Todos los checks pasaron
- ❌ **Fallido**: Al menos un check falló (revisa logs)
- ⏳ **En progreso**: El workflow se está ejecutando

---

## 📊 Status Badge
Para añadir badges de estado en el README.md:

```markdown
![Critical Fixes](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/critical-fixes.yml/badge.svg)
![Compatibility](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml/badge.svg)
![Security](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml/badge.svg)
```

---

## 🔧 Configuración Personalizada

### Cambiar Versiones de Python
Edita la sección `matrix` en el workflow:
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']  # Añade 3.12
```

### Cambiar Ramas Monitoreadas
Edita la sección `on`:
```yaml
on:
  push:
    branches: [ main, develop, staging ]  # Añade staging
  pull_request:
    branches: [ main, develop, staging ]
```

### Cambiar Schedule de Seguridad
Edita la sección `schedule`:
```yaml
schedule:
  - cron: '0 0 * * 0'  # Cambia a diferente hora
```

Formato: `minute hour day month weekday`
- `0 0 * * 0` = Domingo 00:00 UTC
- `0 12 * * *` = Diariamente 12:00 UTC

---

## 📝 Detalles de las Correcciones Críticas

Los workflows validan estas 5 correcciones:

| # | Archivo | Fix | Línea |
|---|---------|-----|-------|
| 1 | `training/trainer.py` | Walk-forward data leakage | 394 |
| 2 | `data/dataset.py` | Volatility calculation (.std not .mean) | 28-52 |
| 3 | `models/layers.py` | Fourier attention determinism | 86-93 |
| 4 | `models/fedformer.py` | Trend projection validation | 160-167 |
| 5 | `models/flows.py` | Log-det jacobian normalization | 105-119 |

Cada workflow valida que estas firmas estén presentes en el código.

---

## 🐛 Troubleshooting

### Workflow no se ejecuta
- Verifica que hayas hecho push a `main` o `develop`
- Revisa que el archivo .yml esté en `.github/workflows/`
- Espera algunos segundos y recarga la página

### Test falla
1. Haz click en la ejecución fallida
2. Expande el paso que falló
3. Lee el log para ver la causa específica
4. Usa la información para debuggear localmente

### Dependencias no encontradas
- Los workflows instalan dependencias automáticamente
- Si falta una dependencia, añádela a `requirements.txt`
- Haz push de cambios y el workflow se re-ejecutará

---

## 📚 Referencias

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Creating status badges](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)

---

## ✅ Validación Local

Para correr las mismas validaciones localmente:

```bash
# Validar correcciones críticas
python -m pytest tests/test_critical_fixes.py -v

# Validar compatibilidad
python tests/validate_fixes.py

# Analizar código
flake8 . --exclude .git,.venv,build,dist
black --check .
isort --check-only .
```

---

**Última actualización:** $(date)
**Status:** ✅ Todos los workflows operacionales
