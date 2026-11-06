from pydantic import BaseModel, Field, HttpUrl


class Chunk(BaseModel):
    id: str
    text: str
    source: str  # file name or URL


class UrlIngestRequest(BaseModel):
    url: HttpUrl


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


class Evidence(BaseModel):
    ref: str  # e.g. S1
    source: str
    text: str


class Review(BaseModel):
    approved: bool
    issues: list[str] = []


class TraceStep(BaseModel):
    node: str
    seconds: float


class ResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    answer: str
    evidence: list[Evidence]
    review: Review
    review_loops: int
    trace: list[TraceStep]
    total_seconds: float


# Structured-output schemas for the LLM
class Plan(BaseModel):
    sub_questions: list[str] = Field(description="1-3 focused sub-questions")


class ReviewOut(BaseModel):
    approved: bool = Field(description="True only if every claim is supported by the evidence")
    issues: list[str] = Field(default_factory=list, description="Unsupported or missing-citation claims")
