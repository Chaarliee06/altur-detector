"""Fetch the pinned official release locally. Never package these files."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

COMMIT = "26b519598a1520cf6306d78902ef5047ae670aa4"
MANIFEST_SHA256 = "4fa5ac3f25f2bc1fbff9a06a89f621fcca30db5f1443bbd164368193f1d7c544"
AUDIO_SHA256 = "41e597bd87dbe76b23b6fffb958f3fbc3761e3fe355a123dff53625f08e936bb"
AUDIO_URL = "https://github.com/alturio/hackmty26/releases/download/v1.0/altur-challenge-audio.zip"


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, destination):
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(destination), url], check=True)


def main():
    root = Path.cwd()
    manifest = root / "manifest.csv"
    if manifest.exists():
        if digest(manifest) != MANIFEST_SHA256:
            raise SystemExit("Existing manifest differs from the pinned official version; it was not overwritten")
    else:
        with tempfile.TemporaryDirectory(prefix="altur-metadata-") as work:
            archive = Path(work) / "source.tar.gz"
            download(f"https://codeload.github.com/alturio/hackmty26/tar.gz/{COMMIT}", archive)
            with tarfile.open(archive) as source:
                source.extractall(Path(work) / "source", filter="data")
            source_root = Path(work) / "source" / f"hackmty26-{COMMIT}"
            if digest(source_root / "manifest.csv") != MANIFEST_SHA256:
                raise SystemExit("Official manifest checksum mismatch")
            shutil.copy2(source_root / "manifest.csv", manifest)
            shutil.copytree(source_root / "turns", root / "turns", dirs_exist_ok=True)
            shutil.copy2(source_root / "README.md", root / "DATASET_README.md")
    if len(list((root / "turns").glob("*.json"))) != 353:
        raise SystemExit("Expected 353 reference turn files; restore the official pinned repo before continuing")
    if len(list((root / "audio").glob("*.wav"))) == 353:
        print("Pinned manifest and local dataset present; phase1 will validate all WAV files")
        return
    with tempfile.TemporaryDirectory(prefix="altur-audio-") as work:
        archive = Path(work) / "audio.zip"
        download(AUDIO_URL, archive)
        if digest(archive) != AUDIO_SHA256:
            raise SystemExit("Release audio checksum mismatch")
        with zipfile.ZipFile(archive) as source:
            members = [item for item in source.infolist() if not item.is_dir()]
            if len(members) != 353:
                raise SystemExit("Unexpected audio archive layout")
            for member in members:
                path = Path(member.filename)
                if len(path.parts) != 2 or path.parts[0] != "audio" or path.suffix != ".wav":
                    raise SystemExit("Unexpected path in official archive")
            source.extractall(root)
    print("353 WAV downloaded and archive SHA-256 verified")


if __name__ == "__main__":
    main()
