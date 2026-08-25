"""
Fast filter for Lichess PGN dumps: games with Stockfish evals + a minimum rating.
Usage:
    python filter_pgn_fast.py input.pgn.zst output.pgn --min-rating 2000
    python filter_pgn_fast.py input.pgn.zst output.pgn --min-rating 2000 --no-eval-required
    python filter_pgn_fast.py input.pgn.zst output.pgn --min-rating 2000 --max-games 500000
"""

import argparse
import io
import re
import sys
import zipfile
import zstandard as zstd

# Precompiled patterns -- header lines look like: [WhiteElo "2431"]
_WHITE_ELO_RE = re.compile(r'\[WhiteElo "(\d+)"\]')
_BLACK_ELO_RE = re.compile(r'\[BlackElo "(\d+)"\]')


def open_pgn_stream(path):
    """Return a text-mode line-iterable file-like object over the PGN content."""
    if path.endswith(".zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh, read_size=1 << 20)
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    elif path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner_name = [n for n in zf.namelist() if n.endswith(".pgn")][0]
        return io.TextIOWrapper(zf.open(inner_name), encoding="utf-8", errors="replace")
    else:
        return open(path, "r", encoding="utf-8", errors="replace")


def iter_games(fh):
    """
    Yield (header_lines, movetext_line) tuples by scanning the stream.

    A Lichess PGN game is: N header lines starting with '[', then a blank
    line, then a single movetext line (or occasionally wrapped across lines),
    then a blank line separating it from the next game. We accumulate until
    we see a movetext line (doesn't start with '[') that's non-empty, then
    treat that as the end of one game's record.
    """
    headers = []
    movetext_parts = []
    in_movetext = False

    for line in fh:
        stripped = line.strip()
        if not stripped:
            if in_movetext and movetext_parts:
                yield headers, " ".join(movetext_parts)
                headers = []
                movetext_parts = []
                in_movetext = False
            continue

        if stripped.startswith("[") and not in_movetext:
            headers.append(stripped)
        else:
            in_movetext = True
            movetext_parts.append(stripped)

    # flush trailing game if file doesn't end with a blank line
    if headers and movetext_parts:
        yield headers, " ".join(movetext_parts)


def passes_filters(headers, movetext, min_rating, require_eval):
    white_elo = black_elo = None
    for h in headers:
        if white_elo is None:
            m = _WHITE_ELO_RE.match(h)
            if m:
                white_elo = int(m.group(1))
                continue
        if black_elo is None:
            m = _BLACK_ELO_RE.match(h)
            if m:
                black_elo = int(m.group(1))

    if white_elo is None or black_elo is None:
        return False
    if white_elo < min_rating or black_elo < min_rating:
        return False

    if require_eval and "%eval" not in movetext:
        return False

    return True


def filter_pgn(input_path, output_path, min_rating, require_eval, max_games=None):
    kept, seen = 0, 0

    with open_pgn_stream(input_path) as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for headers, movetext in iter_games(infile):
            seen += 1

            if seen % 500_000 == 0:
                print(f"...scanned {seen:,} games, kept {kept:,}", file=sys.stderr)

            if not passes_filters(headers, movetext, min_rating, require_eval):
                continue

            outfile.write("\n".join(headers))
            outfile.write("\n\n")
            outfile.write(movetext)
            outfile.write("\n\n")
            kept += 1

            if max_games and kept >= max_games:
                break

    print(f"Done. Scanned {seen:,} games, kept {kept:,} ({kept/max(seen,1)*100:.2f}%).", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to .pgn.zst, .pgn.zip, or plain .pgn")
    ap.add_argument("output", help="Path to write filtered .pgn")
    ap.add_argument("--min-rating", type=int, default=2000, help="Minimum rating for BOTH players")
    ap.add_argument("--no-eval-required", action="store_true", help="Don't require %%eval annotations")
    ap.add_argument("--max-games", type=int, default=None, help="Stop after keeping this many games")
    args = ap.parse_args()

    filter_pgn(
        args.input,
        args.output,
        min_rating=args.min_rating,
        require_eval=not args.no_eval_required,
        max_games=args.max_games,
    )