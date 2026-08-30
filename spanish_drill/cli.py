"""Command line entry point."""
import argparse
import sys

from .config import MAIN_MODEL, VERIFY_MODEL


def build_parser():
    p = argparse.ArgumentParser(
        prog="drill", description="Spanish vocabulary voice drill.")
    p.add_argument("--model", default=MAIN_MODEL,
                   help="local recognition model (default: %(default)s). "
                        "'small' is about 3x faster and measurably less accurate.")
    p.add_argument("--review", action="store_true",
                   help="re-check past answers against a stronger model")
    p.add_argument("--verify-model", default=VERIFY_MODEL,
                   help="model used by --review (default: %(default)s). "
                        "Names starting with 'gpt-' call the OpenAI API.")
    p.add_argument("--devices", action="store_true",
                   help="list the microphones this machine can record from")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.devices:
        from .audio import input_devices
        for index, name in input_devices():
            print(f"  [{index}] {name}")
        return 0

    if args.review:
        from .review import review
        review(args.verify_model)
        return 0

    from PyQt6.QtWidgets import QApplication
    from .ui import Window
    app = QApplication(sys.argv[:1])
    window = Window(args.model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
