"""Quick checks that the command line is wired up and the whole pipeline runs from the raw tickets."""
import pytest

from app import cli


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as err:
        cli.main(["--help"])
    assert err.value.code == 0
    out = capsys.readouterr().out
    for command in ["prepare", "next", "suggest", "interactive", "add", "evaluate", "report"]:
        assert command in out


def test_report_reruns_the_pipeline_and_writes_results(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "raw_interactions.txt").write_text(open("corpus/raw_interactions.txt").read())
    cli.main(["report", "--corpus", str(corpus), "--out", str(tmp_path / "results")])
    evaluation = (tmp_path / "results" / "evaluation.md").read_text()
    assert "baseline" in evaluation and "Kneser-Ney" in evaluation
    assert "Hi <NAME>" in (tmp_path / "results" / "sample_suggestions.md").read_text()
    assert (corpus / "agent_responses.txt").exists()
