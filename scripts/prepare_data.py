"""
scripts/prepare_data.py  (PLACEHOLDER - do not run yet)
------------------------------------------------------
When the team is ready to bring in data, this script will:

  1. Download / copy a set of authentic images   -> data/real/
  2. Download / copy a set of AI-generated images -> data/ai/
  3. Optionally cap the count per class so the hackathon run stays fast.

Nothing is implemented on purpose. Decide the dataset source first
(e.g. a Kaggle real-vs-AI set, or a mix of COCO + a diffusion-model dump),
then fill in the TODOs below.
"""


def main():
    # TODO: pick dataset source(s) and document the license in README.
    # TODO: download archives into a temp folder.
    # TODO: extract and move files into data/real and data/ai.
    # TODO: print final counts per class.
    raise NotImplementedError("prepare_data.py is a placeholder - fill in once the dataset is chosen")


if __name__ == "__main__":
    main()
