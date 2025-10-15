import chess
import chess.engine
import os
import subprocess

def get_supported_backends(lc0_path):
    """Query lc0 --help to detect supported backends."""
    try:
        help_output = subprocess.check_output([lc0_path, "--help"], text=True)
        for line in help_output.splitlines():
            if "VALUES:" in line and "Backend" in line:
                # Parse the allowed backends list
                parts = line.split("VALUES:")[-1].strip()
                return [b.strip() for b in parts.split(",")]
    except Exception as e:
        print(f"Warning: Could not detect backends: {e}")
    return []

def test_lc0():
    lc0_path = "/ada1/u/erickarp/chess/lc0/build/release/lc0"
    weights_path = "/ada1/u/erickarp/chess/skill-compatibility-chess/maia-partner/models/maia-1100.pb.gz"

    try:
        if not os.path.exists(lc0_path):
            raise FileNotFoundError(f"lc0 not found at: {lc0_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at: {weights_path}")

        # Detect which backends are available
        supported = get_supported_backends(lc0_path)
        print(f"Supported lc0 backends: {supported}")

        # Pick blas if available, else fall back to eigen
        backend = "blas" if "blas" in supported else "eigen"

        print(f"Using backend: {backend}")

        # Launch lc0 engine
        transport = chess.engine.SimpleEngine.popen_uci([
            lc0_path,
            f"--weights={weights_path}",
            f"--backend={backend}",
            "--nncache=200000"
        ])

        # Simple test position
        board = chess.Board()
        result = transport.analyse(board, chess.engine.Limit(nodes=100))

        print("Engine test successful!")
        print(f"Best move found: {result['pv'][0]}")

        transport.quit()

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_lc0()
