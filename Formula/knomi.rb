# Homebrew formula for knomi.
#
# This formula is intended to live in the tap repo
# `franjofranjic27/homebrew-knomi` (install via `brew tap franjofranjic27/knomi`
# then `brew install knomi`). It builds knomi and all of its Python runtime
# dependencies into an isolated virtualenv using `Language::Python::Virtualenv`.
#
# ---------------------------------------------------------------------------
# Generating / refreshing the `resource` blocks
# ---------------------------------------------------------------------------
# The 13+ runtime dependencies must be pinned as `resource` blocks below. Do NOT
# write them by hand. Generate them with Homebrew's tooling after setting `url`
# and `sha256` to a real released sdist:
#
#   brew update-python-resources Formula/knomi.rb
#
# (Requires `brew install python-resources` tooling shipped with Homebrew.)
# This reads the `url`, downloads the sdist, resolves the full dependency tree
# from PyPI, and injects one `resource "<name>" do ... end` block per dependency
# right after the `depends_on` lines. Re-run it whenever dependencies change.
#
# ---------------------------------------------------------------------------
# Release automation
# ---------------------------------------------------------------------------
# On every `v*.*.*` tag, the `brew` job in .github/workflows/release-knomi.yaml
# uses `dawidd6/action-homebrew-bump-formula` to bump `url` + `sha256` to the
# newly published PyPI sdist automatically. The PLACEHOLDER values below are only
# used for the very first manual bootstrap of the tap.
# ---------------------------------------------------------------------------

class Knomi < Formula
  include Language::Python::Virtualenv

  desc "Token-efficient document ingestion and RAG connector"
  homepage "https://github.com/franjofranjic27/knomi"
  # PLACEHOLDER — bumped automatically on release to the published PyPI sdist,
  # e.g. https://files.pythonhosted.org/packages/source/k/knomi/knomi-0.1.0.tar.gz
  url "https://files.pythonhosted.org/packages/source/k/knomi/knomi-0.1.0.tar.gz"
  sha256 "PLACEHOLDER" # bumped automatically on release
  license "MIT"

  depends_on "python@3.12"

  # ---- BEGIN generated resources ----
  # Populate with: brew update-python-resources Formula/knomi.rb
  # (leave this marker; the generated `resource` blocks go here)
  # ---- END generated resources ----

  def install
    virtualenv_install_with_resources
  end

  test do
    # The CLI should at least report its help / version without error.
    assert_match "knomi", shell_output("#{bin}/knomi --help")
  end
end
