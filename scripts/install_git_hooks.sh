#!/usr/bin/env bash
# install_git_hooks.sh — FEDformer Probabilistic Time Series Forecasting
#
# Instala (o reinstala) el pre-commit hook del proyecto en .git/hooks/.
# Ejecutar tras clonar el repo o tras recrear el directorio .git/:
#
#   bash scripts/install_git_hooks.sh
#
# El hook replica el pipeline de calidad canónico del proyecto:
#   ruff check --fix → ruff format --check → pylint --errors-only → pytest -q -m "not slow"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
HOOK_FILE="$HOOKS_DIR/pre-commit"

echo "📦  Instalando pre-commit hook en $HOOK_FILE ..."

# Verificar que estamos dentro de un repositorio git
if [ ! -d "$HOOKS_DIR" ]; then
    echo "❌  Error: no se encontró .git/hooks/ en $REPO_ROOT"
    echo "    Asegúrate de ejecutar este script desde dentro del repositorio."
    exit 1
fi

# Escribir el hook directamente (sin depender de un archivo fuente externo rastreado)
cat > "$HOOK_FILE" << 'HOOK_CONTENT'
#!/usr/bin/env bash
# pre-commit hook — FEDformer Probabilistic Time Series Forecasting
# Ejecuta el pipeline de calidad canónico del proyecto antes de cada commit.
# Bloquea el commit si algún check falla.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

# ── 1. Activar venv ─────────────────────────────────────────────────────────
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌  pre-commit: venv no encontrado en .venv/"
    echo "    Crea el entorno con: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

# ── 2. Detectar archivos Python staged ──────────────────────────────────────
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$STAGED_PY" ]; then
    echo "ℹ️   pre-commit: sin archivos Python staged — saltando ruff y pylint, solo pytest"
    RUN_LINT=false
else
    RUN_LINT=true
fi

# ── 3. ruff check --fix ──────────────────────────────────────────────────────
if [ "$RUN_LINT" = true ]; then
    echo "▶  ruff check --fix ..."
    # Capturar lista de archivos antes y después para detectar si --fix modificó algo
    BEFORE=$(git diff --name-only)
    ruff check . --fix
    AFTER=$(git diff --name-only)

    if [ "$BEFORE" != "$AFTER" ]; then
        echo ""
        echo "⚠️   ruff auto-fix aplicó cambios — revisa y re-stagea si es necesario"
        echo "    Archivos modificados por ruff:"
        git diff --name-only | sed 's/^/      /'
        echo ""
        echo "    Comandos sugeridos:"
        echo "      git diff                    # revisar cambios"
        echo "      git add -p                  # stagear interactivamente"
        echo "      git commit                  # reintentar commit"
        exit 1
    fi

    # Verificar que no quedan issues tras el fix
    if ! ruff check . --quiet 2>/dev/null; then
        echo ""
        echo "❌  pre-commit: ruff check encontró problemas que no pudo arreglar automáticamente"
        ruff check .
        exit 1
    fi
fi

# ── 4. ruff format --check ───────────────────────────────────────────────────
if [ "$RUN_LINT" = true ]; then
    echo "▶  ruff format --check ..."
    if ! ruff format . --check --quiet 2>/dev/null; then
        echo ""
        echo "❌  pre-commit: ruff format detectó archivos mal formateados"
        echo "    Ejecuta: ruff format ."
        ruff format . --check
        exit 1
    fi
fi

# ── 5. pylint --errors-only ──────────────────────────────────────────────────
if [ "$RUN_LINT" = true ]; then
    echo "▶  pylint --errors-only models/ training/ data/ utils/ ..."
    if ! pylint --errors-only models/ training/ data/ utils/ 2>/dev/null; then
        echo ""
        echo "❌  pre-commit: pylint encontró errores (E/F) en el código de producción"
        pylint --errors-only models/ training/ data/ utils/
        exit 1
    fi
fi

# ── 6. pytest -q -m "not slow" ───────────────────────────────────────────────
echo "▶  pytest -q -m 'not slow' ..."
if ! pytest -q -m "not slow" --tb=short 2>&1; then
    echo ""
    echo "❌  pre-commit: pytest falló — corrige los tests antes de commitear"
    exit 1
fi

# ── 7. Todo OK ───────────────────────────────────────────────────────────────
echo ""
echo "✅  pre-commit: todos los checks pasaron — commit permitido"
exit 0
HOOK_CONTENT

chmod +x "$HOOK_FILE"

echo "✅  Hook instalado y marcado como ejecutable: $HOOK_FILE"
echo ""
echo "Para verificar que funciona:"
echo "  .git/hooks/pre-commit"
