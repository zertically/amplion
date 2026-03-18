"""
Amplion CLI entry point.

This wrapper exists so the command in `amplion_getting_started.md` works:
`py amplion.py --plain raw.mp4 --examples ref.mp4 --variants 3`
"""

from amplion.amplion import main


if __name__ == "__main__":
    main()

