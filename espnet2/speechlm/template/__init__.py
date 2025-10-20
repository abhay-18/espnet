from .abs_job import AbsJobTemplate
from .speechlm_job import SpeechLMJobTemplate

JOB_TEMPLATES = {
    "speechlm": SpeechLMJobTemplate
}

__all__ = [
    AbsJobTemplate,
    JOB_TEMPLATES,
]