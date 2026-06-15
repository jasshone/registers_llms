from __future__ import annotations

import sys

from run_sink_neuron_pipeline import main


if __name__ == "__main__":
    if "--rescue" not in sys.argv:
        sys.argv.extend(["--rescue", "off"])
    main()
