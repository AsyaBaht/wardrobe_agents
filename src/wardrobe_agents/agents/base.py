"""The agent contract.

An agent is anything with a name and ``run(input) -> output``. The orchestrator
composes agents through this interface alone, so it never needs to know whether
a given agent reasons with Claude, with a rule table, or with both - which is
what lets the weather agent skip its LLM call on an unambiguous day, or lets a
test swap in a stub, without the pipeline noticing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@runtime_checkable
class Agent(Protocol[InputT, OutputT]):
    """Structural type for an agent. Implement it by having ``name`` and ``run``."""

    name: str

    def run(self, payload: InputT) -> OutputT:
        """Execute the agent's single responsibility."""
        ...


AInputT = TypeVar("AInputT")
AOutputT = TypeVar("AOutputT")


class BaseAgent(ABC, Generic[AInputT, AOutputT]):
    """Optional convenience base for the agents in this package.

    Satisfying :class:`Agent` structurally is enough; subclassing only supplies
    the name plumbing and makes the abstract method explicit.
    """

    name: str = "agent"

    @abstractmethod
    def run(self, payload: AInputT) -> AOutputT:
        """Execute the agent's single responsibility."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
