"""Pure unit tests for vra.platform.subtitles (no subprocess, no network)."""

from __future__ import annotations

from pathlib import Path

from vra.platform.subtitles import parse_file

ROLLING_VTT = """\
WEBVTT
Kind: captions
Language: en

NOTE this is a comment block
that spans multiple lines

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.000 --> 00:00:04.000
Hello world
Hello world

00:00:04.000 --> 00:00:06.000
Hello world
and welcome

00:00:06.000 --> 00:00:08.000
and welcome
to the show

00:00:08.000 --> 00:00:10.000
[Music]
{\an8}positioned top
"""


def test_vtt_rolling_dedup(tmp_path: Path) -> None:
    path = tmp_path / "sample.en.vtt"
    path.write_text(ROLLING_VTT, encoding="utf-8")
    transcript = parse_file(path, 24000)

    assert transcript.source == "subtitle"
    # Fully duplicated cues merge away; rolled repeated lines (identical to the
    # previously accepted line) are dropped; [Music] speech marker is kept.
    texts = [seg.text for seg in transcript.segments]
    assert texts == [
        "Hello world",
        "and welcome",
        "to the show",
        "[Music] positioned top",
    ]
    assert [seg.start_s for seg in transcript.segments] == [0.0, 4.0, 6.0, 8.0]
    assert transcript.segments[1].end_s == 6.0
    assert transcript.text == (
        "Hello world and welcome to the show [Music] positioned top"
    )


def test_vtt_timestamps_milliseconds(tmp_path: Path) -> None:
    path = tmp_path / "t.vtt"
    path.write_text(
        "WEBVTT\n\n00:00.500 --> 00:01.250\nhi\n", encoding="utf-8"
    )
    transcript = parse_file(path, 1000)
    assert transcript.segments[0].start_s == 0.5
    assert transcript.segments[0].end_s == 1.25


def test_srt_parsing(tmp_path: Path) -> None:
    path = tmp_path / "sample.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,500\nFirst line\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nSecond\nline\n\n"
        "3\n00:00:06,000 --> 00:00:08,000\nSecond\nline\n",
        encoding="utf-8",
    )
    transcript = parse_file(path, 24000)
    assert len(transcript.segments) == 2
    first, second = transcript.segments
    assert first.start_s == 1.0
    assert first.end_s == 3.5
    assert first.text == "First line"
    assert second.text == "Second line"


def test_max_chars_clamps_mid_segment(tmp_path: Path) -> None:
    path = tmp_path / "clamp.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\naaaaa\n\n"
        "00:00:01.000 --> 00:00:02.000\nbbbbbbbbbb\n\n"
        "00:00:02.000 --> 00:00:03.000\ncccc\n",
        encoding="utf-8",
    )
    transcript = parse_file(path, 8)
    assert transcript.text == "aaaaa bb"
    assert [len(s.text) for s in transcript.segments] == [5, 2]
    assert len(transcript.text) <= 8


def test_empty_and_garbage_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty.vtt"
    empty.write_text("WEBVTT\n", encoding="utf-8")
    transcript = parse_file(empty, 100)
    assert transcript.text == ""
    assert transcript.segments == ()

    garbage = tmp_path / "garbage.vtt"
    garbage.write_text("not a subtitle at all\n", encoding="utf-8")
    assert parse_file(garbage, 100).segments == ()

    assert parse_file(empty, 0).text == ""


def test_markup_stripped_but_speech_tags_kept(tmp_path: Path) -> None:
    path = tmp_path / "markup.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000 align:start position:0%\n"
        "<v Speaker>bold <b>text</b></v> [Applause]\n\n",
        encoding="utf-8",
    )
    transcript = parse_file(path, 1000)
    assert transcript.segments[0].text == "bold text [Applause]"
