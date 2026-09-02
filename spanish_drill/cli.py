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
    p.add_argument("--conjugations", action="store_true",
                   help="with --review, re-check the conjugation drill's own "
                        "log against its own schedule. Without it, --review "
                        "would repair those answers into the vocabulary "
                        "tracker, which is the one thing that mode never "
                        "writes to.")
    p.add_argument("--placement", action="store_true",
                   help="rapid placement test: right twice is known, wrong once "
                        "goes to the learning pile")
    p.add_argument("--category", default=None,
                   help="restrict the drill to one part of speech, e.g. 'verb'. "
                        "Use --categories to see what is available.")
    p.add_argument("--categories", action="store_true",
                   help="list the parts of speech in the deck")
    p.add_argument("--serve", action="store_true",
                   help="drill from a phone: serves a typed drill over HTTP. "
                        "Everything stays on this machine.")
    p.add_argument("--port", type=int, default=8765,
                   help="port for --serve (default: %(default)s)")
    p.add_argument("--devices", action="store_true",
                   help="list the microphones this machine can record from")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.categories:
        from .deck import categories, load_deck
        deck = load_deck()
        print(f"  {'all':<14} {len(deck)}")
        for pos in categories(deck):
            print(f"  {pos:<14} {sum(1 for c in deck if c.pos == pos)}")
        return 0

    if args.category:
        from .progress import Progress
        p = Progress.load()
        p.category = args.category
        p.save()
        print(f"drill restricted to: {args.category}")

    if args.devices:
        from .audio import input_devices
        for index, name in input_devices():
            print(f"  [{index}] {name}")
        return 0

    if args.serve:
        from .serve import serve
        return serve(port=args.port)

    if args.review:
        from .review import review
        if args.conjugations:
            from .answers import AnswerLog
            from .config import CONJUGATION_LOG
            from .paradigm import ConjugationProgress
            from .progress import Progress
            review(args.verify_model, log=AnswerLog(path=CONJUGATION_LOG),
                   progress=ConjugationProgress.open(Progress.load()))
        else:
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
