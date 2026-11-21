from __future__ import annotations

from sphura.domain.errors import NotFoundError
from sphura.pipeline.preferences import KNOWN_TRAITS, Rating, resolve_trait, traits_of
from sphura.ports.repositories import CandidateRepository, ClipRepository, FeedbackRepository


class FeedbackService:
    def __init__(
        self,
        clips: ClipRepository,
        candidates: CandidateRepository,
        feedback: FeedbackRepository,
    ) -> None:
        self._clips = clips
        self._candidates = candidates
        self._feedback = feedback

    def rate_clips(self, project_id: str, good: list[int], bad: list[int]) -> list[str]:
        """Rate clips by the number in their file name (01_, 02_, ...). Returns titles."""
        moments = {c.id: c for c in self._candidates.list_for_project(project_id)}
        by_rank = {
            c.rank: c
            for c in self._clips.list_for_project(project_id)
            if c.rank and c.candidate_id in moments
        }
        rated = []
        for number, is_good in [(n, True) for n in good] + [(n, False) for n in bad]:
            clip = by_rank.get(number)
            if clip is None or clip.candidate_id not in moments:
                raise NotFoundError(f"no clip #{number} in this project (see: sphura clips)")
            moment = moments[clip.candidate_id]
            self._feedback.rate(project_id, Rating(moment.id, is_good, traits_of(moment)))
            rated.append(f"{'good' if is_good else 'bad '} #{number} {clip.title}")
        return rated

    def rate_moment(self, project_id: str, candidate_id: str, good: bool) -> None:
        moment = next(
            (c for c in self._candidates.list_for_project(project_id) if c.id == candidate_id),
            None,
        )
        if moment is None:
            raise NotFoundError("moment not found")
        self._feedback.rate(project_id, Rating(moment.id, good, traits_of(moment)))

    def ratings(self, project_id: str) -> dict[str, bool]:
        ids = {c.id for c in self._candidates.list_for_project(project_id)}
        return {r.candidate_id: r.good for r in self._feedback.all() if r.candidate_id in ids}

    def set_taste(self, words: list[str], likes: bool | None) -> list[str]:
        """--like / --dislike / --forget. Returns the traits changed."""
        changed = []
        for word in words:
            trait = resolve_trait(word)
            if trait is None:
                known = ", ".join(sorted(k for k in KNOWN_TRAITS if " " not in k or "clips" in k))
                raise NotFoundError(f"don't know '{word}'. Try one of: {known}")
            self._feedback.set_override(trait, likes)
            changed.append(word)
        return changed
