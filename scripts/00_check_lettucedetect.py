from __future__ import annotations

from lettucedetect.models.inference import HallucinationDetector


def main() -> None:
    detector = HallucinationDetector(
        method="transformer",
        model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",
    )
    spans = detector.predict(
        context=["Paris is the capital of France."],
        question="What is the capital of France?",
        answer="The capital of France is Berlin.",
        output_format="spans",
    )
    print(spans)
    if not spans:
        raise RuntimeError("LettuceDetect returned no spans on a trivial contradiction.")


if __name__ == "__main__":
    main()
